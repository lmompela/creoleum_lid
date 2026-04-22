#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ALPHA_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def normalize_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    # Handle "id<TAB>text" rows (as in French dataset)
    if "\t" in line:
        line = line.split("\t", 1)[1].strip()
    return line


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def is_word(token: str) -> bool:
    return bool(ALPHA_RE.search(token))


def load_labeled_samples(path: Path) -> tuple[list[str], list[str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [normalize_line(line) for line in raw]
    lines = [line for line in lines if line]

    sentences: list[str] = []
    tokens: list[str] = []

    for line in lines:
        for sent in split_sentences(line):
            if len(sent) >= 3:
                sentences.append(sent.lower())
            for tok in tokenize(sent):
                tok_l = tok.lower()
                if is_word(tok_l) and len(tok_l) >= 2:
                    tokens.append(tok_l)

    return sentences, tokens


def build_pipeline_for_sentences() -> any:
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            lowercase=True,
            min_df=2,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        ),
    )


def build_pipeline_for_tokens() -> any:
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            min_df=2,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        ),
    )


def evaluate_token_gold(model, eval_csv: Path) -> None:
    if not eval_csv.exists():
        print(f"Skipping external eval (missing file): {eval_csv}")
        return
    df = pd.read_csv(eval_csv)
    if "token" not in df.columns or "label" not in df.columns:
        print(f"Skipping external eval (token/label columns not found): {eval_csv}")
        return
    df = df.dropna(subset=["token", "label"])
    X = df["token"].astype(str).str.lower().tolist()
    y_true = (
        df["label"]
        .astype(str)
        .str.strip()
        .map({"Creole": "gcf", "Not_Creole": "fra"})
        .tolist()
    )
    y_pred = model.predict(X)
    print("\n=== Token Gold Eval ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production LID models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = cfg_path.parents[1]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    train_creole = (repo_root / cfg["paths"]["train_creole"]).resolve()
    train_french = (repo_root / cfg["paths"]["train_french"]).resolve()
    eval_csv = (repo_root / cfg["paths"]["eval_token_csv"]).resolve()
    model_out = (repo_root / cfg["paths"]["model_out"]).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training data (gcf): {train_creole}")
    print(f"Training data (fra): {train_french}")

    gcf_sent, gcf_tok = load_labeled_samples(train_creole)
    fra_sent, fra_tok = load_labeled_samples(train_french)

    Xs = gcf_sent + fra_sent
    ys = ["gcf"] * len(gcf_sent) + ["fra"] * len(fra_sent)
    Xt = gcf_tok + fra_tok
    yt = ["gcf"] * len(gcf_tok) + ["fra"] * len(fra_tok)

    print(f"Sentence samples: gcf={len(gcf_sent)} fra={len(fra_sent)}")
    print(f"Token samples:    gcf={len(gcf_tok)} fra={len(fra_tok)}")

    test_size = float(cfg.get("training", {}).get("test_size", 0.2))
    seed = int(cfg.get("training", {}).get("random_seed", 42))

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        Xs, ys, test_size=test_size, random_state=seed, stratify=ys
    )
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        Xt, yt, test_size=test_size, random_state=seed, stratify=yt
    )

    sent_model = build_pipeline_for_sentences()
    tok_model = build_pipeline_for_tokens()

    sent_model.fit(Xs_train, ys_train)
    tok_model.fit(Xt_train, yt_train)

    ys_pred = sent_model.predict(Xs_test)
    yt_pred = tok_model.predict(Xt_test)

    print("\n=== Sentence Holdout ===")
    print(f"Accuracy: {accuracy_score(ys_test, ys_pred):.4f}")
    print(classification_report(ys_test, ys_pred, digits=4))

    print("\n=== Token Holdout ===")
    print(f"Accuracy: {accuracy_score(yt_test, yt_pred):.4f}")
    print(classification_report(yt_test, yt_pred, digits=4))

    evaluate_token_gold(tok_model, eval_csv)

    bundle = {
        "sentence_model": sent_model,
        "token_model": tok_model,
        "classes": ["fra", "gcf"],
        "inference": cfg.get("inference", {"token_weight": 0.7, "neighbor_weight": 0.15}),
    }
    joblib.dump(bundle, model_out)
    print(f"\nSaved model bundle: {model_out}")


if __name__ == "__main__":
    main()
