
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "klue/roberta-base"
DATA_PATH = Path("dataset_v0.csv")
OUTPUT_DIR = Path("model_output")
SEED = 42
MAX_LENGTH = 128


def load_data() -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV에는 {required} 열이 필요합니다.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates(subset=["text"])

    labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    return df, label2id, id2label


def split_data(df: pd.DataFrame) -> DatasetDict:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["labels"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["labels"],
    )

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            frame[["text", "labels"]].reset_index(drop=True),
            preserve_index=False,
        )

    return DatasetDict(
        {
            "train": to_dataset(train_df),
            "validation": to_dataset(valid_df),
            "test": to_dataset(test_df),
        }
    )


def main() -> None:
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df, label2id, id2label = load_data()
    datasets = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = datasets.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    trainer.train()

    test_output = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids

    print("\n=== 테스트 평가 ===")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[id2label[i] for i in range(len(id2label))],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metadata = {
        "model_name": MODEL_NAME,
        "label2id": label2id,
        "id2label": id2label,
        "max_length": MAX_LENGTH,
        "dataset_size": len(df),
    }
    with (final_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {final_dir.resolve()}")


if __name__ == "__main__":
    main()
