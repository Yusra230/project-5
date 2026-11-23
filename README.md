# Project 5 – Live Training Dashboard

A Streamlit dashboard that trains a small neural network on the Wisconsin Breast Cancer dataset, displays training/validation loss, diagnoses underfitting or overfitting, tracks experiments, and demonstrates predictions on held-out records.

## Objectives
- Load 569 records with 30 numeric measurements; malignant is encoded as class `1`.
- Split data into stratified 60% train, 20% validation, 20% test partitions.
- Fit feature scaling on training rows only and reuse it for validation/test.
- Train with configurable hidden size, dropout, weight decay, learning rate and early stopping.
- Display loss curves, experiment history, CSV export and a held-out patient example.

## Project structure
- `app.py` – Streamlit user interface and live chart callback.
- `trainer.py` – reusable PyTorch training engine, diagnosis and evaluation.
- `run_experiments.py` – reproducible four-run experiment script.
- `artifacts/` – generated experiment table, report and learning-curve image.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app.py
```

## Generate reproducible results
```bash
python run_experiments.py
```
This creates `artifacts/experiments.csv`, `artifacts/training_curves.png` and `artifacts/REPORT.md`.

## Diagnosis rules
- Underfitting: final training loss > 0.30.
- Overfitting: final validation loss > 1.30 × the best validation loss.
- Otherwise: healthy.

These are the explicit teaching rules from the assignment brief, not universal clinical/model-quality thresholds.

## Limitations and responsible use
- The Wisconsin dataset is small and tabular; it is not a live hospital feed or medical imaging dataset.
- A single random split does not establish generalisation to other hospitals or populations.
- The diagnosis thresholds are simplified educational heuristics.
- The model can make false-negative and false-positive predictions.
- No calibration, external validation, clinical workflow integration or regulatory review is provided.
- **Never use this project to diagnose, treat, or make decisions about real patients.** A qualified clinician must review any clinical decision.
