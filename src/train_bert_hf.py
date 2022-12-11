#!/usr/bin/env python
#
# Train a vision learner on mnist using huggingface models
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

mnist = load_dataset("mnist", split="train[:50]")
mnist = mnist.train_test_split(test_size=0.2)

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
    ixs = ixs[:20]
    train_sentences.append(DELIM.join(ixs))


test_sentences = []
for i in tqdm(range(mnist['test'].shape[0])):
    x = np.ravel(mnist['test'][i]['pixel_values'].numpy())
    shuffle_order = skutils.shuffle(range(len(x)))
    shuffle_sort_order = np.argsort(-x[shuffle_order], kind='stable')
    ixs = img_loc_labels[shuffle_order][shuffle_sort_order]
    #ixs = ixs[x[shuffle_order][shuffle_sort_order] > 0]
    ixs = ixs[:20]
    test_sentences.append(DELIM.join(ixs))


mnist_text = dsets.DatasetDict({
    'train': dsets.Dataset.from_list(
        [{'text': train_sentences[i], 'label': mnist['train'][i]['label']}
         for i in range(mnist['train'].shape[0])]
    ),
    'test': dsets.Dataset.from_list(
        [{'text': test_sentences[i], 'label': mnist['test'][i]['label']}
         for i in range(mnist['test'].shape[0])]
    )
})

################################################################################
## Train Model
################################################################################

tokenizer = tfs.AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = tfs.AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# we will need to add tokens to our tokenizer and give reasonable
# initial embeddings.
#
# See: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
''
tokenizer.add_tokens(img_loc_labels.tolist())

tokenizer(mnist_text['train'][0]['text'])

model.resize_token_embeddings(len(tokenizer))


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

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_mnist_text = mnist_text.map(preprocess_function, batched=True)
data_collator = tfs.DataCollatorWithPadding(tokenizer=tokenizer)


training_args = tfs.TrainingArguments(
    output_dir="calc/mnist_text",
    learning_rate=1e-2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)


trainer = tfs.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mnist_text["test"],
    eval_dataset=tokenized_mnist_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

for param in model.base_model.transformer.parameters():
    param.requires_grad = False

trainer.train()



text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
classifier = tfs.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer)
classifier(text)

i = 1
mnist_text['test'][i]['label']
classifier(mnist_text['test'][i]['text'])

i = 2
mnist_text['test'][i]['label']
classifier(mnist_text['test'][i]['text'])

i = 3
mnist_text['test'][i]['label']
classifier(mnist_text['test'][i]['text'])

i = 4
mnist_text['test'][i]['label']
classifier(mnist_text['test'][i]['text'])

i = 5
mnist_text['test'][i]['label']
classifier(mnist_text['test'][i]['text'])









checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = tfs.AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

