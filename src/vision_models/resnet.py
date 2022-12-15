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
import pandas as pd

import argparse

################################################################################
## Parse Command Line Arguments
################################################################################

parser = argparse.ArgumentParser(
    prog = 'ResNetModel',
    description = 'run resnet model on image datasets'
)

parser.add_argument('-d', '--dataset')
parser.add_argument('-n', '--num_training_samples')
parser.add_argument('-t', '--num_epochs')
parser.add_argument('-o', '--training_history_outfile')

args = parser.parse_args()

################################################################################
## Hyperparameters
################################################################################

accuracy = evl.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


################################################################################
## Load Data
################################################################################

mnist = load_dataset(args.dataset)
mnist['train'] = load_dataset(args.dataset,
    split="train[:{}]".format(args.num_training_samples))

labels = mnist["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


################################################################################
## Train Model
################################################################################

image_processor = tfs.AutoImageProcessor.from_pretrained(
    "microsoft/resnet-34")

_transforms = tt.Compose([
    tt.ToTensor()
])

def transforms(examples):
    if "image" in examples:
        examples["pixel_values"] = [_transforms(img.convert('RGB')) for img in examples["image"]]
        del examples["image"]
    if "img" in examples:
        examples["pixel_values"] = [_transforms(img.convert('RGB')) for img in examples["img"]]
        del examples["img"]
    return examples



mnist = mnist.with_transform(transforms)
data_collator = tfs.DefaultDataCollator()


model = tfs.AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-34",
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


training_args = tfs.TrainingArguments(
    output_dir="calc/{}".format(args.dataset),
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=int(args.num_epochs),
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = tfs.Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=mnist["train"],
    eval_dataset=mnist["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

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

plot_df.to_csv(args.training_history_outfile, index=False)

print('All done!')
