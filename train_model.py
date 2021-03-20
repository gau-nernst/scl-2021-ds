import numpy as np
import pandas as pd
import torch
import logging
import json

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_model_and_tokenizer(model_checkpoint, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    return model, tokenizer

def tokenize_and_align_labels(texts, tags, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, is_split_into_words=True)

    labels = []

    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class AddressDataset(torch.utils.data.Dataset):
    def __init__(self, df, tag2id, tokenizer):
        tokens = df["tokens"].to_list()
        labels = df["label"].to_list()
        tags = [[tag2id[x] for x in sample] for sample in labels]
        
        self.encodings = tokenize_and_align_labels(tokens, tags, tokenizer)
    
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["labels"])

class ComputeMetrics:
    def __init__(self, id2tag):
        self.id2tag = id2tag
    
    def compute(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1":f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

def main(name):
    logging.info("Start of training")
    
    train_df = pd.read_json("train_processed.json")
    val_df = pd.read_json("val_processed.json")

    unique_tags = set(tag for label in train_df["label"].to_list() for tag in label)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    with open(f"tag2id_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"tag2id": tag2id, "id2tag": id2tag}, f)

    model, tokenizer = get_model_and_tokenizer("xlm-roberta-base", len(unique_tags))

    train_dataset = AddressDataset(train_df, tag2id, tokenizer)
    val_dataset = AddressDataset(val_df, tag2id, tokenizer)

    compute_metrics = ComputeMetrics(id2tag).compute

    training_args = TrainingArguments(
        output_dir=f'./results_{name}',
        save_steps=1000,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_{name}",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # trainer.evaluate()
    trainer.save_model(f"./model_{name}")


if __name__ == "__main__":
    name = "run4"
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', 
        level=logging.INFO, 
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ])
    main(name)