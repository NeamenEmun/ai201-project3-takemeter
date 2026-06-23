import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABELS = [
    'Game reaction',
    'Analysis / strategy',
    'News / injury update',
    'Meme / opinion',
]

class NBADataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = {label: i for i, label in enumerate(LABELS)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row['text']
        label = self.label2id[row['label']]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=128)
        return {**encoding, 'labels': label}


def load_data(path: Path) -> List[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['label'].strip()]
    return rows


def split_dataset(rows: List[dict], seed: int = 42):
    if len(rows) < 10:
        random.seed(seed)
        random.shuffle(rows)
        n = len(rows)
        train_n = int(0.8 * n)
        val_n = int(0.1 * n)
        return rows[:train_n], rows[train_n:train_n + val_n], rows[train_n + val_n:]

    labels = [row['label'] for row in rows]
    min_label_count = min(Counter(labels).values())
    if min_label_count >= 2:
        train_rows, temp_rows, train_labels, temp_labels = train_test_split(
            rows,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
        val_rows, test_rows = train_test_split(
            temp_rows,
            temp_labels,
            test_size=0.5,
            random_state=seed,
            stratify=temp_labels,
        )
        return train_rows, val_rows, test_rows

    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    return rows[:train_n], rows[train_n:train_n + val_n], rows[train_n + val_n:]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds)
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def train(path: str, output_dir: str):
    data = load_data(Path(path))
    train_rows, val_rows, test_rows = split_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_ds = NBADataset(train_rows, tokenizer)
    val_ds = NBADataset(val_rows, tokenizer)
    test_ds = NBADataset(test_rows, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(LABELS)
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print('Training finished. Running evaluation on test set...')
    predictions = trainer.predict(test_ds)
    metrics = compute_metrics(predictions)
    print('Test accuracy:', metrics['accuracy'])
    print('Test macro F1:', metrics['macro_f1'])
    print('Confusion matrix:')
    print(np.array(metrics['confusion_matrix']))
    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train/evaluate TakeMeter classifier')
    parser.add_argument('--data', default='nba_dataset_submission.csv')
    parser.add_argument('--output', default='model_output')
    args = parser.parse_args()
    train(args.data, args.output)
