#!/usr/bin/env python
#
# Train a vision learner on mnist using huggingface models from scratch
#

from datasets import load_dataset
import transformers as tfs
import torch
import torchvision.transforms as tt
import evaluate as evl
import numpy as np
import datasets as dsets

from tqdm import tqdm
import sklearn.utils as skutils

import plotnine as pn
import pandas as pd

################################################################################
## Hyperparameters
################################################################################

accuracy = evl.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


DELIM = ' '

################################################################################
## Load Data
################################################################################

mnist = load_dataset("fashion_mnist")

labels = mnist["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


_transforms = tt.Compose([
    tt.ToTensor()
])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
    del examples["image"]
    return examples


mnist = mnist.with_transform(transforms)

# now convert to sequences using np.argsort with random tiebreaks
# shuffle -> stable sort

img_loc_shape = mnist['train'][0]['pixel_values'].numpy().shape

img_loc_labels = np.array(
    ['i'+str(i)+'r'+str(r)+'c'+str(c) 
    for i in range(img_loc_shape[0])
    for r in range(img_loc_shape[1])
    for c in range(img_loc_shape[2])])

train_sentences = []
for i in tqdm(range(mnist['train'].shape[0])):
    x = np.ravel(mnist['train'][i]['pixel_values'].numpy())
    shuffle_order = skutils.shuffle(range(len(x)))
    shuffle_sort_order = np.argsort(-x[shuffle_order], kind='stable')
    ixs = img_loc_labels[shuffle_order][shuffle_sort_order]
    #ixs = ixs[x[shuffle_order][shuffle_sort_order] > 0]
    ixs = ixs[:256]
    train_sentences.append(DELIM.join(ixs))


test_sentences = []
for i in tqdm(range(mnist['test'].shape[0])):
    x = np.ravel(mnist['test'][i]['pixel_values'].numpy())
    shuffle_order = skutils.shuffle(range(len(x)))
    shuffle_sort_order = np.argsort(-x[shuffle_order], kind='stable')
    ixs = img_loc_labels[shuffle_order][shuffle_sort_order]
    #ixs = ixs[x[shuffle_order][shuffle_sort_order] > 0]
    ixs = ixs[:256]
    test_sentences.append(DELIM.join(ixs))


mnist_text = dsets.DatasetDict({
    'train': dsets.Dataset.from_list(
        [{'text': train_sentences[i], 'label': mnist['train'][i]['label']}
         #for i in range(mnist['train'].shape[0])]
         for i in range(1000)]
    ),
    'test': dsets.Dataset.from_list(
        [{'text': test_sentences[i], 'label': mnist['test'][i]['label']}
         for i in range(mnist['test'].shape[0])]
         #for i in range(1000)]
    )
})

################################################################################
## Train Model
################################################################################

tokenizer = tfs.AutoTokenizer.from_pretrained("distilbert-base-uncased")

config = tfs.DistilBertConfig()
model_raw = tfs.DistilBertModel(config)

model = tfs.AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

params = model_raw.state_dict()
params = {'distilbert.' + k:params[k] for k in params.keys()}
for k in model.state_dict().keys():
    if k not in params:
        params[k] = model.state_dict()[k]


# reset transformer weights to initialization conditions
model.load_state_dict(params)

# we will need to add tokens to our tokenizer and give reasonable
# initial embeddings.
#
# See: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

tokenizer.add_tokens(img_loc_labels.tolist())
model.resize_token_embeddings(len(tokenizer))

"""
n_new = len(img_loc_labels)

params = model.state_dict()
embeddings = params['distilbert.embeddings.word_embeddings.weight']
pre_expansion_embeddings = embeddings[:-n_new,:]
mu = torch.mean(pre_expansion_embeddings, dim=0)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

new_embeddings = torch.stack(tuple((dist.sample() for _ in range(n_new))), dim=0)
embeddings[-n_new:,:] = new_embeddings
params['distilbert.embeddings.word_embeddings.weight'][-n_new:,:] = new_embeddings
model.load_state_dict(params)
"""


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_mnist_text = mnist_text.map(preprocess_function, batched=True)
data_collator = tfs.DataCollatorWithPadding(tokenizer=tokenizer)


training_args = tfs.TrainingArguments(
    output_dir="calc/mnist_text",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    load_best_model_at_end=True,
    push_to_hub=False,
)


trainer = tfs.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mnist_text["train"],
    eval_dataset=tokenized_mnist_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


for param in model.base_model.transformer.parameters():
    param.requires_grad = False


trainer.train()

################################################################################
## Plot Training Curves
################################################################################


epoch2loss = {}
epoch2eval_loss = {}
epoch2eval_acc = {}
for d in trainer.state.log_history:
    if 'epoch' in d and 'loss' in d:
        epoch2loss[d['epoch']] = d['loss']
    if 'epoch' in d and 'eval_loss' in d:
        epoch2eval_loss[d['epoch']] = d['eval_loss']
    if 'epoch' in d and 'eval_accuracy' in d:
        epoch2eval_acc[d['epoch']] = d['eval_accuracy']

value_type_map = {
    'loss': epoch2loss,
    'eval_loss': epoch2eval_loss,
    'eval_accuracy': epoch2eval_acc
}

plot_df = pd.DataFrame([{
    'epoch': e,
    'value': value_type_map[t][e],
    'type': t
} for e in epoch2loss.keys()
  for t in value_type_map.keys()])

print(plot_df)

plot_df.to_csv('./calc/training_history_fromscratch.csv', index=False)

print('All done!')
