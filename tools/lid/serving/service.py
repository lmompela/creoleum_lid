import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ALPHA_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)
PUNCT_RE = re.compile(r"^[^\w\s]+$", flags=re.UNICODE)


def _is_word(token: str) -> bool:
    return bool(ALPHA_RE.search(token))


def _is_punct(token: str) -> bool:
    return bool(PUNCT_RE.match(token))


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _normalize_probs(raw_classes, raw_probs) -> dict[str, float]:
    d = {cls: float(p) for cls, p in zip(raw_classes, raw_probs)}
    # enforce expected keys
    for key in ("fra", "gcf"):
        d.setdefault(key, 0.0)
    s = d["fra"] + d["gcf"]
    if s <= 0:
        return {"fra": 0.5, "gcf": 0.5}
    return {"fra": d["fra"] / s, "gcf": d["gcf"] / s}


def _combine_probs(token_probs: dict[str, float], sent_probs: dict[str, float], token_weight: float) -> dict[str, float]:
    eps = 1e-12
    tw = max(0.0, min(1.0, token_weight))
    sw = 1.0 - tw
    fra = np.exp(tw * np.log(token_probs["fra"] + eps) + sw * np.log(sent_probs["fra"] + eps))
    gcf = np.exp(tw * np.log(token_probs["gcf"] + eps) + sw * np.log(sent_probs["gcf"] + eps))
    s = fra + gcf
    if s <= 0:
        return {"fra": 0.5, "gcf": 0.5}
    return {"fra": float(fra / s), "gcf": float(gcf / s)}


@lru_cache(maxsize=1)
def _load_bundle():
    env_path = os.getenv("LID_MODEL_PATH")
    if env_path:
        model_path = Path(env_path).resolve()
    else:
        model_path = (Path(__file__).resolve().parents[3] / "models" / "lid_model.joblib").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"LID model not found: {model_path}")
    return joblib.load(model_path)


def run_lid(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return {"language": "fra", "confidence": 0.5, "per_token": []}

    bundle = _load_bundle()
    sentence_model = bundle["sentence_model"]
    token_model = bundle["token_model"]
    inf_cfg = bundle.get("inference", {})
    token_weight = float(inf_cfg.get("token_weight", 0.7))
    neighbor_weight = float(inf_cfg.get("neighbor_weight", 0.15))

    sent_raw = sentence_model.predict_proba([text])[0]
    sent_probs = _normalize_probs(sentence_model.classes_, sent_raw)
    tokens = _tokenize(text)

    per_token_probs: list[dict[str, float]] = []
    word_indexes: list[int] = []

    for idx, token in enumerate(tokens):
        if _is_word(token):
            raw = token_model.predict_proba([token.lower()])[0]
            tok_probs = _normalize_probs(token_model.classes_, raw)
            probs = _combine_probs(tok_probs, sent_probs, token_weight=token_weight)
            word_indexes.append(idx)
        elif _is_punct(token):
            probs = dict(sent_probs)
        else:
            probs = dict(sent_probs)
        per_token_probs.append(probs)

    # light neighborhood smoothing for lexical tokens only
    bw = max(0.0, min(0.45, neighbor_weight))
    if bw > 0 and word_indexes:
        smoothed = [dict(p) for p in per_token_probs]
        for idx in word_indexes:
            nbrs = []
            if idx - 1 >= 0 and _is_word(tokens[idx - 1]):
                nbrs.append(per_token_probs[idx - 1])
            if idx + 1 < len(tokens) and _is_word(tokens[idx + 1]):
                nbrs.append(per_token_probs[idx + 1])
            if nbrs:
                fra_n = sum(n["fra"] for n in nbrs) / len(nbrs)
                gcf_n = sum(n["gcf"] for n in nbrs) / len(nbrs)
                fra = (1.0 - bw) * per_token_probs[idx]["fra"] + bw * fra_n
                gcf = (1.0 - bw) * per_token_probs[idx]["gcf"] + bw * gcf_n
                s = fra + gcf
                smoothed[idx] = {"fra": fra / s, "gcf": gcf / s}
        per_token_probs = smoothed

    per_token = []
    sent_fra = []
    sent_gcf = []
    for token, probs in zip(tokens, per_token_probs):
        lang = "gcf" if probs["gcf"] >= probs["fra"] else "fra"
        conf = probs[lang]
        per_token.append({"token": token, "lang": lang, "conf": round(float(conf), 4)})
        if _is_word(token):
            sent_fra.append(probs["fra"])
            sent_gcf.append(probs["gcf"])

    if sent_fra:
        fra_score = float(np.mean(sent_fra))
        gcf_score = float(np.mean(sent_gcf))
    else:
        fra_score = sent_probs["fra"]
        gcf_score = sent_probs["gcf"]

    language = "gcf" if gcf_score >= fra_score else "fra"
    confidence = gcf_score if language == "gcf" else fra_score

    return {
        "language": language,
        "confidence": round(float(confidence), 4),
        "per_token": per_token,
    }
