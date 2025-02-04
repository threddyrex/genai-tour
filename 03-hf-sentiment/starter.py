
#
# Example program that uses huggingface transformers to train a sentiment analysis model.
#
# From: https://huggingface.co/docs/transformers/tasks/sequence_classification
#


from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datetime import datetime


def Log(msg):
    print("----------------------------------------")
    print(datetime.now().strftime("%H:%M:%S"), msg)
    print("----------------------------------------")


Log("load_dataset")
imdb = load_dataset('imdb')


Log("AutoTokenizer.from_pretrained")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


Log("map")
tokenized_imdb = imdb.map(preprocess_function, batched=True)

Log("DataCollatorWithPadding")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


Log("evaluate.load")
accuracy = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


Log("AutoModelForSequenceClassification.from_pretrained")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

Log("TrainingArguments")
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

Log("Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# This is projected to take the better part of a day to run.
# Need to find a way to speed this up.
Log("trainer.train()")
trainer.train()

