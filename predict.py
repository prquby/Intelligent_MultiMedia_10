
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model_output/final")
MAX_LENGTH = 128


def predict(text: str) -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {MODEL_DIR}\n"
            "먼저 python train_classifier.py 를 실행하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    ranked = sorted(
        [
            (model.config.id2label[i], float(probs[i]))
            for i in range(len(probs))
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n입력: {text}")
    print(f"최종 판단: {ranked[0][0]} ({ranked[0][1]:.2%})")
    print("\n전체 확률:")
    for label, prob in ranked:
        print(f"- {label}: {prob:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="분류할 한국어 문장")
    args = parser.parse_args()
    predict(" ".join(args.text))


if __name__ == "__main__":
    main()
