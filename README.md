# Credit Card Fraud Detection with ANN and LSTM

An educational machine-learning project that detects potentially fraudulent card transactions. The Streamlit dashboard compares an Artificial Neural Network (ANN) with a Long Short-Term Memory (LSTM) network using the same complete training and held-out test datasets.

## Models and results

The reported scores below use the full held-out test split and a probability threshold of 0.50. Fraud is rare in this dataset, so accuracy should be considered together with precision, recall, and F1.

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| ANN | 97.96% | 13.17% | 76.78% | 0.225 | 0.601 |
| LSTM | 94.16% | 5.41% | 85.78% | 0.102 | 0.488 |

ANN produces a better balance of precision and recall in these results. LSTM catches a larger share of fraud cases, but also flags more legitimate transactions. Results may change with different data, features, training settings, or decision thresholds.

## Dataset

The project expects a ZIP archive containing `fraudTrain.csv` and `fraudTest.csv`. The preparation script keeps transaction time, category, amount, gender, city population, job, and the fraud label. It excludes card numbers, names, addresses, transaction IDs, and precise location fields.

The prepared splits used for the reported results contain:

- Training: 1,296,675 transactions
- Held-out test: 555,719 transactions
- Total: 1,852,394 transactions

The archive itself is not included in this repository. Obtain the dataset from your authorized source and follow its license and sharing requirements.

## Setup on Windows

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\prepare_dataset.py "C:\path\to\CC_dataset.zip"
python scripts\train_fraud_model.py
python -m streamlit run app.py
```

Replace `C:\path\to\CC_dataset.zip` with the actual location of your dataset archive. Streamlit prints the local URL to open, usually `http://localhost:8501`.

Training evaluates all rows in the prepared train and test splits. It can take several minutes depending on the computer. To train only the LSTM while reusing the saved ANN results, run:

```powershell
python scripts\train_fraud_model.py --reuse-ann
```

## Dashboard pages

- **Overview:** dataset sizes and class balance.
- **Model Comparison:** train and held-out test metrics, plus ROC and precision-recall curves.
- **ANN & LSTM Lab:** ANN learning history and LSTM training details.
- **Data Explorer:** filtered previews of train and test records.
- **Final Report:** test-set metrics, confusion matrix, curves, and fraud patterns.

## LSTM sequence limitation

The prepared dataset has no card or customer identifier. The LSTM therefore reads each transaction with the previous nine transactions in chronological order within its split. These are market-wide transaction windows, not histories for individual cards. A dataset with a stable card identifier would be needed to build and evaluate per-card sequences.

## Project structure

```text
app.py                       Streamlit dashboard
scripts/prepare_dataset.py   Prepare modeling CSV files from the ZIP
scripts/train_fraud_model.py Train and evaluate ANN and LSTM
data/processed/              Prepared train and test CSVs (generated locally)
models/                      Saved model weights and evaluation outputs
requirements.txt             Python dependencies
```

## Disclaimer

This is an academic demonstration using a simulated dataset. The results do not establish performance on real banking transactions and must not be used to make actual financial decisions.
