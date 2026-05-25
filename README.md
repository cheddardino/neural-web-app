# DressCheck — Deployment Guide

## Project Structure
```
dress-code-app/
├── api/
│   └── predict.py        ← Python inference endpoint
├── models/
│   ├── svm_model.pkl     ← trained SVM (copy from Midterm Dataset)
│   └── scaler.pkl        ← StandardScaler (copy from Midterm Dataset)
├── index.html            ← single-page frontend
├── requirements.txt
└── vercel.json
```

## Step 1 — Copy model files
Copy these from your Midterm Dataset folder into `models/`:
- `svm_model.pkl`  (if named differently, update the path in api/predict.py)
- `scaler.pkl`

## Step 2 — Push to GitHub
```bash
git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/dress-code-app.git
git push -u origin main
```

## Step 3 — Deploy on Vercel
1. Go to vercel.com → New Project → Import your GitHub repo
2. Framework Preset: **Other**
3. Click Deploy — done.

## ⚠ Important Notes
- Vercel free tier has a 250MB function size limit.
  scikit-learn + scikit-image + numpy ≈ ~120MB. Should be fine.
- Cold starts may take ~5–10 seconds on first load (model loading).
- The SVM model filename in `api/predict.py` is `svm_model.pkl`.
  Update line 20 if your file has a different name.
