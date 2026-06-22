from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model_output/final")
TEST_CSV = Path("final_test_dataset_500.csv")

OUTPUT_ALL = Path("evaluation_predictions.csv")
OUTPUT_ERRORS = Path("evaluation_errors.csv")
OUTPUT_CONFUSION = Path("confusion_matrix.csv")
OUTPUT_BINARY_CONFUSION = Path("binary_confusion_matrix.csv")

MAX_LENGTH = 128
BATCH_SIZE = 16

RISK_LABELS = {"PERSONAL_INFO", "CREDENTIAL", "INTERNAL_INFO"}


def load_test_data() -> pd.DataFrame:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"테스트 파일을 찾을 수 없습니다: {TEST_CSV.resolve()}\n"
            "테스트 CSV를 이 스크립트와 같은 폴더에 두세요."
        )

    df = pd.read_csv(TEST_CSV)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("테스트 CSV에는 text, label 열이 필요합니다.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"학습된 모델을 찾을 수 없습니다: {MODEL_DIR.resolve()}\n"
            "먼저 train_classifier.py를 실행하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    confidences = []
    texts = df["text"].tolist()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

        for i, pred_id in enumerate(pred_ids.tolist()):
            predictions.append(model.config.id2label[pred_id])
            confidences.append(float(probs[i, pred_id].cpu()))

    result = df.copy()
    result["prediction"] = predictions
    result["confidence"] = confidences
    result["correct"] = result["label"] == result["prediction"]
    return result


def evaluate_multiclass(df: pd.DataFrame) -> None:
    labels = ["SAFE", "PERSONAL_INFO", "CREDENTIAL", "INTERNAL_INFO"]
    y_true = df["label"]
    y_pred = df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print("1. 4개 라벨 분류 평가")
    print("=" * 70)
    print(f"Accuracy    : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\n라벨별 Precision / Recall / F1")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0,
    ))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    matrix_df.to_csv(OUTPUT_CONFUSION, encoding="utf-8-sig")
    print("Confusion Matrix:")
    print(matrix_df)


def evaluate_binary(df: pd.DataFrame) -> None:
    y_true = df["label"].map(lambda x: "UNSAFE" if x in RISK_LABELS else "SAFE")
    y_pred = df["prediction"].map(lambda x: "UNSAFE" if x in RISK_LABELS else "SAFE")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["UNSAFE"],
        average="binary",
        pos_label="UNSAFE",
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=["SAFE", "UNSAFE"])
    tn, fp, fn, tp = matrix.ravel()

    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    binary_accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print("2. SAFE / UNSAFE 통합 평가")
    print("=" * 70)
    print(f"Binary Accuracy     : {binary_accuracy:.4f} ({binary_accuracy:.2%})")
    print(f"UNSAFE Precision    : {precision:.4f} ({precision:.2%})")
    print(f"UNSAFE Recall       : {recall:.4f} ({recall:.2%})")
    print(f"UNSAFE F1           : {f1:.4f}")
    print(f"False Negative Rate : {fnr:.4f} ({fnr:.2%})")
    print(f"False Positive Rate : {fpr:.4f} ({fpr:.2%})")

    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_SAFE", "actual_UNSAFE"],
        columns=["pred_SAFE", "pred_UNSAFE"],
    )
    matrix_df.to_csv(OUTPUT_BINARY_CONFUSION, encoding="utf-8-sig")
    print("\nBinary Confusion Matrix:")
    print(matrix_df)


def save_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")
    errors = df[~df["correct"]].copy()
    errors.to_csv(OUTPUT_ERRORS, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("3. 결과 파일")
    print("=" * 70)
    print(f"전체 예측 결과 : {OUTPUT_ALL.resolve()}")
    print(f"오답 목록       : {OUTPUT_ERRORS.resolve()}")
    print(f"4분류 혼동행렬  : {OUTPUT_CONFUSION.resolve()}")
    print(f"이진 혼동행렬   : {OUTPUT_BINARY_CONFUSION.resolve()}")
    print(f"오답 개수       : {len(errors)} / {len(df)}")

    if not errors.empty:
        print("\n오답 예시 최대 10개:")
        for _, row in errors.head(10).iterrows():
            print("-" * 70)
            print(f"문장   : {row['text']}")
            print(f"정답   : {row['label']}")
            print(f"예측   : {row['prediction']}")
            print(f"확신도 : {row['confidence']:.2%}")


def main() -> None:
    df = load_test_data()
    result = run_predictions(df)
    print(f"평가 문장 수: {len(result)}")
    evaluate_multiclass(result)
    evaluate_binary(result)
    save_outputs(result)


if __name__ == "__main__":
    main()
