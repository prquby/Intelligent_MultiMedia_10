
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model_output/final")
MAX_LENGTH = 128
SAFE_THRESHOLD = 0.60


@dataclass
class SegmentResult:
    text: str
    label: str
    confidence: float
    safe_probability: float

    @property
    def is_safe(self) -> bool:
        return self.label == "SAFE" and self.safe_probability >= SAFE_THRESHOLD


def split_into_segments(text: str) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text.strip())
    if not normalized:
        return []

    # 문장 뒤에 띄어쓰기가 없어도 분리합니다.
    # 예: "작성해줘.사용자가" -> "작성해줘." / "사용자가"
    #
    # 마침표(.)는 IP 주소(192.0.2.48)나 소수(31.8)를 자르지 않도록
    # 다음 문자가 한글일 때만 경계로 봅니다.
    # 물음표/느낌표는 한글 또는 영문 문장이 바로 이어져도 분리합니다.
    parts = re.split(
        r"\n+"
        r"|(?<=[.。])(?=[가-힣])"
        r"|(?<=[!?！？])(?=[가-힣A-Za-z])"
        r"|(?<=[.!?。！？])\s+"
        r"|(?<=;)\s*",
        normalized,
    )

    return [part.strip() for part in parts if part.strip()]


class RiskClassifierGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("한국어 AI 입력 위험 분석기")
        self.root.geometry("820x680")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        self.build_ui()
        self.load_model()

    def build_ui(self):
        main = tk.Frame(self.root, padx=12, pady=12)
        main.pack(fill="both", expand=True)

        input_frame = tk.Frame(main, bd=3, relief="solid")
        input_frame.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(
            input_frame,
            text="입력칸",
            font=("Malgun Gothic", 18, "bold"),
            pady=8
        ).pack()

        self.input_text = tk.Text(
            input_frame,
            height=12,
            font=("Malgun Gothic", 12),
            wrap="word"
        )
        self.input_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        middle = tk.Frame(main)
        middle.pack(fill="x", pady=(0, 12))

        tk.Label(
            middle,
            text="작동 버튼",
            font=("Malgun Gothic", 13, "bold")
        ).pack(side="left", padx=(0, 12))

        tk.Button(
            middle,
            text="문장별 분석",
            font=("Malgun Gothic", 12, "bold"),
            width=12,
            command=self.run_analysis
        ).pack(side="left")

        tk.Button(
            middle,
            text="결과 복사",
            font=("Malgun Gothic", 12),
            width=10,
            command=self.copy_output
        ).pack(side="left", padx=(10, 0))

        tk.Button(
            middle,
            text="초기화",
            font=("Malgun Gothic", 12),
            width=10,
            command=self.clear_all
        ).pack(side="left", padx=(10, 0))

        self.status_label = tk.Label(
            middle,
            text="모델 확인 중...",
            font=("Malgun Gothic", 10)
        )
        self.status_label.pack(side="right")

        output_frame = tk.Frame(main, bd=3, relief="solid")
        output_frame.pack(fill="both", expand=True)

        tk.Label(
            output_frame,
            text="문제가 있는 문장",
            font=("Malgun Gothic", 18, "bold"),
            pady=8
        ).pack()

        self.output_text = tk.Text(
            output_frame,
            height=15,
            font=("Malgun Gothic", 12),
            wrap="word"
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.output_text.tag_configure(
            "danger",
            foreground="#B00020",
            font=("Malgun Gothic", 12, "bold")
        )
        self.output_text.tag_configure(
            "safe",
            foreground="#1B5E20",
            font=("Malgun Gothic", 12, "bold")
        )
        self.output_text.tag_configure(
            "header",
            font=("Malgun Gothic", 13, "bold")
        )
        self.output_text.tag_configure(
            "detail",
            foreground="#444444"
        )

    def load_model(self):
        if not MODEL_DIR.exists():
            self.status_label.config(text="모델 없음")
            messagebox.showerror(
                "모델 없음",
                f"학습된 모델 폴더를 찾을 수 없습니다.\n\n{MODEL_DIR.resolve()}"
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.to(self.device)
            self.model.eval()
            self.status_label.config(text=f"모델 준비 완료 ({self.device.type})")
        except Exception as exc:
            self.status_label.config(text="모델 로드 실패")
            messagebox.showerror("모델 로드 실패", str(exc))

    def classify_segments(self, segments: list[str]) -> list[SegmentResult]:
        safe_label_id = self.model.config.label2id["SAFE"]
        results = []

        for start in range(0, len(segments), 16):
            batch = segments[start:start + 16]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)

            predicted_ids = torch.argmax(probabilities, dim=-1)

            for index, predicted_id in enumerate(predicted_ids.tolist()):
                label = self.model.config.id2label[predicted_id]
                results.append(
                    SegmentResult(
                        text=batch[index],
                        label=label,
                        confidence=float(probabilities[index, predicted_id].cpu()),
                        safe_probability=float(probabilities[index, safe_label_id].cpu())
                    )
                )

        return results

    def run_analysis(self):
        source_text = self.input_text.get("1.0", "end").strip()

        if not source_text:
            messagebox.showwarning("입력 필요", "입력할 내용을 넣어주세요.")
            return

        if self.model is None or self.tokenizer is None:
            messagebox.showerror("모델 오류", "모델이 로드되지 않았습니다.")
            return

        segments = split_into_segments(source_text)

        try:
            self.status_label.config(text=f"{len(segments)}개 문장 분석 중...")
            self.root.update_idletasks()

            results = self.classify_segments(segments)
            danger_results = [result for result in results if not result.is_safe]

            self.output_text.delete("1.0", "end")

            if not danger_results:
                self.output_text.insert(
                    "end",
                    "문제가 있는 문장이 발견되지 않았습니다.\n",
                    "safe"
                )
                self.output_text.insert(
                    "end",
                    f"\n총 {len(results)}개 문장을 분석했습니다.",
                    "detail"
                )
            else:
                self.output_text.insert(
                    "end",
                    f"총 {len(results)}개 중 {len(danger_results)}개가 위험 후보입니다.\n\n",
                    "header"
                )

                for number, result in enumerate(danger_results, start=1):
                    self.output_text.insert(
                        "end",
                        f"{number}. {result.text}\n",
                        "danger"
                    )
                    self.output_text.insert(
                        "end",
                        f"   예측 확신도: {result.confidence:.2%}\n",
                        "detail"
                    )
                    self.output_text.insert(
                        "end",
                        f"   설명: {self.label_explanation(result.label)}\n\n",
                        "detail"
                    )

            self.status_label.config(
                text=f"분석 완료: 위험 후보 {len(danger_results)}개"
            )

        except Exception as exc:
            self.status_label.config(text="분석 실패")
            messagebox.showerror("분석 실패", str(exc))

    @staticmethod
    def label_explanation(label: str) -> str:
        explanations = {
            "PERSONAL_INFO": "개인을 식별하거나 개인과 연결될 수 있는 정보가 포함된 것으로 판단되었습니다.",
            "CREDENTIAL": "비밀번호, API 키, 토큰, 접속 문자열 등 인증정보가 포함된 것으로 판단되었습니다.",
            "INTERNAL_INFO": "계약 조건, 단가, 미공개 계획, 내부 운영 정보 등이 포함된 것으로 판단되었습니다."
        }
        return explanations.get(label, "위험 정보로 분류되었습니다.")

    def copy_output(self):
        result = self.output_text.get("1.0", "end").strip()

        if not result:
            messagebox.showinfo("복사", "복사할 결과가 없습니다.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(result)
        self.root.update()
        messagebox.showinfo("복사", "결과를 복사했습니다.")

    def clear_all(self):
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.status_label.config(text="입력 대기 중")


if __name__ == "__main__":
    root = tk.Tk()
    RiskClassifierGUI(root)
    root.mainloop()
