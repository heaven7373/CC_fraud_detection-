# Credit Card Fraud Detection

An academic binary-classification project using the supplied simulated credit-card transaction dataset. It compares an Artificial Neural Network (ANN) with a chronological Long Short-Term Memory (LSTM) network on the same train/test split.

## Dataset

The supplied `CC_dataset.zip` contains `fraudTrain.csv` and `fraudTest.csv`. Import it with:

```powershell
python scripts/prepare_dataset.py "C:\path\to\CC_dataset.zip"
```

The preparation script writes `data/processed/fraud_train.csv` and `fraud_test.csv`. It keeps transaction time, category, amount, gender, city population, job, and the fraud label; names, card numbers, addresses, transaction IDs, and precise location fields are excluded. The archive itself is not copied into the repository.

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py "C:\path\to\CC_dataset.zip"
python scripts/train_fraud_model.py
python -m streamlit run app.py
```

The trainer uses every row in `fraud_train.csv` and evaluates both models on every row in `fraud_test.csv`. The ANN has 128, 64, and 32 ReLU neurons, class weights, and early stopping. The LSTM reads ten chronologically ordered transaction rows per window. Use `python scripts/train_fraud_model.py --reuse-ann` to retain the saved ANN scores and train only the new LSTM. The dashboard includes train/test comparison tables and charts for accuracy, precision, recall, F1, ROC–AUC, and average precision, plus ROC/precision–recall curves and confusion matrices. The 0.5 threshold is an initial reference point and can be tuned for a particular alert-cost tradeoff.

## Limitations

The prepared dataset omits card/customer IDs, so the LSTM models chronological market-wide windows and cannot learn per-card histories. The archive is simulated and does not demonstrate production performance on real bank transactions. Results depend on the dataset, training sample, features, and threshold. Scores are for educational review, not automated financial decisions.
