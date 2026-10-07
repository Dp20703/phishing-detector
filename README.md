# Phishing URL Detector

A small end-to-end machine learning app. Paste a link and it estimates how
likely the URL is to be phishing, using only the text of the URL. It shows a
risk score, a three-level verdict (Likely Safe / Uncertain / Suspicious) and
the rule-based warning signs found in the link.

**Stack:** Python, pandas, scikit-learn, FastAPI, React, TypeScript, Vite, Tailwind CSS

**Live demo:** _add your Vercel link here_ &nbsp;|&nbsp; **API:** _add your Render link here_

## Problem statement

Phishing links trick people into giving away passwords and payment details,
and they are hard to spot by eye. This project trains a classifier on the
structure of a URL (length, digits, subdomains, suspicious words, brand
look-alikes, randomness) and serves it through a REST API and a web UI so a
user can check a link before clicking it.

## How it works

```text
React + TypeScript  ──POST /predict──▶  FastAPI  ──▶  features.py ──▶ Random Forest
   (Vercel)                            (Render)         (shared)       (joblib file)
```

1. `backend/features.py` normalizes the URL (removes `https://`, `www.`,
   fragments) and computes 16 numeric features. **The same file is used for
   training and for the API**, so the model always sees identically computed inputs.
2. `ml/train.py` trains and compares two models and saves the better one.
3. `backend/main.py` loads the saved model once at startup and serves `POST /predict`.
4. The React app calls the API and shows the result.

## Dataset

Public labeled URL dataset (`urlset.csv`, about 96k URLs, roughly 50/50
legitimate and phishing). Download it from Kaggle (search for "phishing URLs
urlset") and place it at `ml/data/urlset.csv`. The file is not committed.

Only the raw URL and the label are used. The dataset's other columns are
precomputed from the web pages and cannot be computed for a new URL.

## Method and honest evaluation

- **Split by website, not by row.** The 80/20 split groups URLs by their
  registered domain, so the test set contains only sites the model has never
  seen. A plain random split puts the same site in both sets and inflates scores.
- **Model selection without touching the test set.** Logistic Regression and
  Random Forest were compared with grouped 3-fold cross-validation on the
  training data only. The test set was used once, at the end.
- **Legit homepages were added to the training split.** The dataset contains
  no legitimate bare-domain URLs, so the original model called `google.com`
  100% phishing. Homepages of legitimate training sites were added to the
  training split only (never to the test split).
- **Random Forest size is limited** (`max_depth=18`, `min_samples_leaf=3`):
  the saved model is 12.5 MB instead of about 200 MB.

### Results (test set, unseen sites)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.848 | 0.904 | 0.757 | 0.824 |
| **Random Forest** (selected) | **0.890** | **0.926** | **0.834** | **0.877** |

Random Forest confusion matrix (rows = actual, columns = predicted;
order: legitimate, phishing):

```text
              pred legit   pred phishing
actual legit      9330           596
actual phish      1478          7402
```

Precision and recall matter here because the two mistakes are not equal: a
missed phishing link (false negative) is more dangerous than a false alarm,
and accuracy alone hides that trade-off. Recall for phishing is 0.834, so
about 1 in 6 phishing URLs in the test set is missed.

Top features by Random Forest importance: URL length, suspicious words outside
the domain, character entropy, digit count and host length. Importance only
describes what this model learned from this dataset, not what makes a URL
dangerous in general.

Re-run `python ml/train.py` to regenerate these numbers (saved to `ml/metrics.json`).

## Limitations (please read)

- **The dataset is old and has a shape bias.** Its legitimate URLs are mostly
  older, short pages, while its phishing URLs are long and messy. The model
  partly learned "long and complicated means phishing". On a small hand-made
  set of modern URLs (`ml/eval/real_world_urls.csv`: 41 legitimate links and 20
  phishing-style patterns, **not a benchmark**), it flagged 7 of 41 legitimate
  links as phishing (mostly deep links on large sites such as GitHub and MDN,
  and sign-in pages) and caught 20 of 20 phishing-style patterns. Expect false
  alarms on long, legitimate links.
- **The middle band is intentional.** Probabilities between 0.35 and 0.65 are
  shown as "Uncertain" instead of forcing a verdict.
- **URL text only.** The app never visits the page, checks reputation lists, or
  inspects content. A clean-looking URL can still be malicious, and a flagged
  URL can be safe.
- **The listed signals are rule-based**, simple checks written by hand. They
  are not the model's internal reasoning.
- **Not a security product.** Do not rely on it to decide whether a link is safe.

Possible future work: train on newer data with realistic legitimate deep links,
add page-level features, and calibrate probabilities.

## Run locally

Requires Python 3.12 and Node 20+.

```bash
# 1. Backend
cd backend
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload      # http://127.0.0.1:8000/docs

# 2. Frontend (new terminal)
cd frontend
npm install
cp .env.example .env           # points to the local API
npm run dev                    # http://localhost:5173
```

### Retrain the model (optional)

```bash
# from the project root, with ml/data/urlset.csv in place
pip install -r backend/requirements.txt matplotlib jupyter
python ml/train.py
```

The saved model must be loaded with the same scikit-learn version it was
trained with (pinned in `backend/requirements.txt`). If you change versions, retrain.

## API

`POST /predict`

```json
{ "url": "https://example.com/login" }
```

```json
{
  "url": "https://example.com/login",
  "label": "legitimate",
  "risk_level": "low",
  "probability": 0.12,
  "reasons": ["No obvious suspicious patterns in the URL text"],
  "model": "Random Forest"
}
```

`probability` is the model's phishing-class output on its training data, not a
guarantee. `GET /health` returns `{"status": "ok"}`.

## Deploy

**Backend (Render, Web Service):**
- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variable: `ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`

**Frontend (Vercel):**
- Root directory: `frontend`
- Environment variable: `VITE_API_URL=https://<your-render-service>.onrender.com`

On Render's free tier the service sleeps when idle, so the first request can
take up to a minute. The UI shows a message if the first call fails.

## Project structure

```text
phishing-detector/
├── backend/
│   ├── main.py              FastAPI app
│   ├── features.py          shared URL feature extraction + signals
│   ├── requirements.txt
│   └── model/phishing_model.joblib
├── ml/
│   ├── train.py             training + evaluation
│   ├── notebook.ipynb       exploratory data analysis
│   ├── metrics.json         results from the last training run
│   ├── eval/real_world_urls.csv
│   └── data/urlset.csv      (download separately, not committed)
└── frontend/                React + Vite + TypeScript + Tailwind
```
