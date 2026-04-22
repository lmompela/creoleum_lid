# Creoleum LID

Production-oriented language identification for French (`fra`) vs Martinican/Guadeloupean Creole (`gcf`).

## Features

- Sentence-level classifier (character n-grams)
- Token-level classifier (character n-grams)
- Context-aware inference (sentence prior + token evidence + neighbor smoothing)
- Drop-in service API:
  - `language`
  - `confidence`
  - `per_token`

## Repository Layout

```text
.
├── configs/
│   └── train_config.yaml
├── data/
│   └── README.md
├── models/
├── tools/
│   └── lid/
│       ├── serving/service.py
│       └── training/train.py
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place training/eval files under `data/`:

- `data/creole_dataset.txt`
- `data/french_dataset.txt`
- `data/lid_eval_tokens.csv` (optional token-level eval with `token,label`)

## Train

```bash
python3 -m tools.lid.training.train --config configs/train_config.yaml
```

This writes `models/lid_model.joblib`.

## Inference API

```bash
python3 -c "from tools.lid.serving.service import run_lid; print(run_lid('Comment tu te rappelles ?'))"
```

By default, service loads `models/lid_model.joblib`.  
Override with:

```bash
export LID_MODEL_PATH=/absolute/path/to/lid_model.joblib
```

## Example Output

```python
{
  "language": "fra",
  "confidence": 0.9658,
  "per_token": [
    {"token": "Comment", "lang": "fra", "conf": 0.9706},
    {"token": "tu", "lang": "fra", "conf": 0.9410},
    ...
  ]
}
```
