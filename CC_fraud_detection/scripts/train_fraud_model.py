"""Train ANN and chronological LSTM models on the complete fraud dataset splits."""
import json
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
NUMERIC = ["amt", "hour", "day", "weekday", "city_pop"]
CATEGORICAL = ["category", "gender", "job"]
SEQ_LEN = 10
THRESHOLD = 0.5


def engineer_features(frame):
    out = frame.copy()
    out["hour"] = out.trans_date_trans_time.dt.hour
    out["day"] = out.trans_date_trans_time.dt.day
    out["weekday"] = out.trans_date_trans_time.dt.dayofweek
    return out[NUMERIC + CATEGORICAL]


def chart_points(*arrays, n=2000):
    indices = np.linspace(0, len(arrays[0]) - 1, min(n, len(arrays[0]))).astype(int)
    return [np.asarray(values)[indices].tolist() for values in arrays]


def evaluate(name, labels, proba, train_metrics, n_train, n_test, train_time):
    pred = proba >= THRESHOLD
    fpr, tpr, _ = roc_curve(labels, proba)
    pr_p, pr_r, _ = precision_recall_curve(labels, proba)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    return {
        "model": name, "threshold": THRESHOLD, "train_time_sec": round(train_time, 2),
        "accuracy": float(accuracy_score(labels, pred)),
        "roc_auc": float(roc_auc_score(labels, proba)),
        "average_precision": float(average_precision_score(labels, proba)),
        "fraud_recall": float(recall_score(labels, pred, zero_division=0)),
        "fraud_precision": float(precision_score(labels, pred, zero_division=0)),
        "fraud_f1": float(f1_score(labels, pred, zero_division=0)),
        "training_rows": int(n_train), "test_rows": int(n_test),
        "test_fraud_rows": int(np.sum(labels)), "train_metrics": train_metrics,
        "confusion_matrix": cm.tolist(),
        "roc_curve": dict(zip(("fpr", "tpr"), chart_points(fpr, tpr))),
        "precision_recall_curve": dict(zip(("precision", "recall"), chart_points(pr_p, pr_r))),
        "classification_report": classification_report(labels, pred, output_dict=True, zero_division=0),
    }


def basic_metrics(labels, proba):
    pred = proba >= THRESHOLD
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "fraud_precision": float(precision_score(labels, pred, zero_division=0)),
        "fraud_recall": float(recall_score(labels, pred, zero_division=0)),
        "fraud_f1": float(f1_score(labels, pred, zero_division=0)),
    }


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
    ])


class TransactionLSTM(nn.Module):
    def __init__(self, category_count, gender_count, job_count, numeric_count):
        super().__init__()
        self.category_embedding = nn.Embedding(category_count, 8)
        self.gender_embedding = nn.Embedding(gender_count, 3)
        self.job_embedding = nn.Embedding(job_count, 16)
        self.lstm = nn.LSTM(numeric_count + 27, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)

    def forward(self, numeric, categorical):
        embedded = torch.cat((
            numeric,
            self.category_embedding(categorical[:, :, 0]),
            self.gender_embedding(categorical[:, :, 1]),
            self.job_embedding(categorical[:, :, 2]),
        ), dim=2)
        _, (hidden, _) = self.lstm(embedded)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


def encoded_features(train_features, test_features):
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_features[NUMERIC]).astype(np.float32)
    test_numeric = scaler.transform(test_features[NUMERIC]).astype(np.float32)
    train_cats, test_cats, vocab_sizes = [], [], []
    mappings = {}
    for column in CATEGORICAL:
        values = train_features[column].fillna("<missing>").astype(str)
        vocab = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
        mappings[column] = vocab
        train_cats.append(values.map(vocab).to_numpy(dtype=np.int64))
        test_values = test_features[column].fillna("<missing>").astype(str)
        test_cats.append(test_values.map(vocab).fillna(0).to_numpy(dtype=np.int64))
        vocab_sizes.append(len(vocab) + 1)
    return (train_numeric, np.stack(train_cats, axis=1),
            test_numeric, np.stack(test_cats, axis=1), scaler, mappings, vocab_sizes)


def window_indices(start, end, length, device):
    rows = torch.arange(start, end, device=device)
    offsets = torch.arange(length - 1, -1, -1, device=device)
    return (rows[:, None] - offsets[None, :]).clamp_min(0)


def predict_lstm(model, numeric, categorical, batch_size, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            end = min(start + batch_size, len(numeric))
            idx = window_indices(start, end, SEQ_LEN, device)
            logits = model(numeric[idx], categorical[idx])
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-ann", action="store_true", help="reuse the already-trained ANN and its saved full-split scores")
    args = parser.parse_args()
    print("Loading complete train and test splits...")
    train = pd.read_csv(DATA / "fraud_train.csv", parse_dates=["trans_date_trans_time"])
    test = pd.read_csv(DATA / "fraud_test.csv", parse_dates=["trans_date_trans_time"])
    X_train, X_test = engineer_features(train), engineer_features(test)
    y_train, y_test = train.is_fraud.to_numpy(dtype=np.int64), test.is_fraud.to_numpy(dtype=np.int64)
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    weights = np.where(y_train == 1, len(y_train) / (2 * positive), len(y_train) / (2 * negative))
    MODELS.mkdir(exist_ok=True)
    scores = {}
    predictions = {"actual_fraud": y_test.tolist()}

    metrics_path = MODELS / "fraud_metrics.json"
    predictions_path = MODELS / "fraud_test_predictions.csv"
    if args.reuse_ann and metrics_path.exists() and predictions_path.exists():
        previous_scores = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous_predictions = pd.read_csv(predictions_path)
        if "ANN" not in previous_scores or "ann_probability" not in previous_predictions or len(previous_predictions) != len(test):
            raise ValueError("Saved ANN results do not match the current dataset. Run without --reuse-ann to train it again.")
        scores["ANN"] = previous_scores["ANN"]
        predictions["ann_probability"] = previous_predictions["ann_probability"].tolist()
        print("\n[1/2] Reusing saved ANN model scores and predictions.")
    else:
        print("\n[1/2] Training ANN...")
        ann = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                                   alpha=0.0003, batch_size=1024, learning_rate_init=0.001,
                                   max_iter=35, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=6, random_state=42)),
        ])
        started = time.time()
        ann.fit(X_train, y_train, clf__sample_weight=weights)
        ann_test = ann.predict_proba(X_test)[:, 1]
        ann_train = ann.predict_proba(X_train)[:, 1]
        ann_time = time.time() - started
        ann_result = evaluate("Artificial Neural Network (MLP)", y_test, ann_test, basic_metrics(y_train, ann_train), len(train), len(test), ann_time)
        ann_clf = ann.named_steps["clf"]
        ann_result["architecture"] = list(ann_clf.hidden_layer_sizes)
        ann_result["epochs"] = int(ann_clf.n_iter_)
        ann_result["training_history"] = {"loss": ann_clf.loss_curve_, "validation_accuracy": ann_clf.validation_scores_}
        scores["ANN"] = ann_result
        predictions["ann_probability"] = ann_test.tolist()
        joblib.dump(ann, MODELS / "fraud_ann.joblib")
        print(f"   ANN done in {ann_time:.1f}s · test recall {ann_result['fraud_recall']:.3f}")

    print("\n[2/2] Training chronological LSTM (10 transactions per window)...")
    started = time.time()
    train_order = np.argsort(train.trans_date_trans_time.to_numpy(), kind="stable")
    test_order = np.argsort(test.trans_date_trans_time.to_numpy(), kind="stable")
    X_train_lstm = X_train.iloc[train_order].reset_index(drop=True)
    X_test_lstm = X_test.iloc[test_order].reset_index(drop=True)
    y_train_lstm = y_train[train_order]
    y_test_lstm = y_test[test_order]
    train_num, train_cat, test_num, test_cat, scaler, mappings, vocab_sizes = encoded_features(X_train_lstm, X_test_lstm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    train_num = torch.from_numpy(train_num).to(device)
    train_cat = torch.from_numpy(train_cat).to(device)
    test_num = torch.from_numpy(test_num).to(device)
    test_cat = torch.from_numpy(test_cat).to(device)
    labels = torch.from_numpy(y_train_lstm.astype(np.float32)).to(device)
    lstm = TransactionLSTM(vocab_sizes[0], vocab_sizes[1], vocab_sizes[2], len(NUMERIC)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    epochs, batch_size = 5, 2048
    history = {"loss": []}
    lstm.train()
    for epoch in range(epochs):
        order = torch.randperm(len(train), device=device)
        total_loss, seen = 0.0, 0
        for start in range(0, len(train), batch_size):
            batch_rows = order[start:start + batch_size]
            idx = window_indices_for_rows(batch_rows, len(train), SEQ_LEN)
            optimizer.zero_grad(set_to_none=True)
            logits = lstm(train_num[idx], train_cat[idx])
            loss = criterion(logits, labels[batch_rows])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_rows)
            seen += len(batch_rows)
        mean_loss = total_loss / seen
        history["loss"].append(mean_loss)
        print(f"   Epoch {epoch + 1}/{epochs} · loss {mean_loss:.4f}")
    lstm_train_sorted = predict_lstm(lstm, train_num, train_cat, batch_size, device)
    lstm_test_sorted = predict_lstm(lstm, test_num, test_cat, batch_size, device)
    lstm_train = np.empty_like(lstm_train_sorted)
    lstm_test = np.empty_like(lstm_test_sorted)
    lstm_train[train_order] = lstm_train_sorted
    lstm_test[test_order] = lstm_test_sorted
    lstm_time = time.time() - started
    lstm_result = evaluate("Long Short-Term Memory (LSTM)", y_test, lstm_test, basic_metrics(y_train, lstm_train), len(train), len(test), lstm_time)
    lstm_result["sequence_length"] = SEQ_LEN
    lstm_result["training_history"] = history
    lstm_result["device"] = str(device)
    scores["LSTM"] = lstm_result
    predictions["lstm_probability"] = lstm_test.tolist()
    torch.save({"state_dict": lstm.state_dict(), "vocab_sizes": vocab_sizes, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "mappings": mappings, "sequence_length": SEQ_LEN}, MODELS / "fraud_lstm.pt")
    print(f"   LSTM done in {lstm_time:.1f}s · test recall {lstm_result['fraud_recall']:.3f} · device {device}")

    (MODELS / "fraud_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(MODELS / "fraud_test_predictions.csv", index=False)
    print("\nTraining complete. Full test split comparison:")
    for name, result in scores.items():
        print(f"{name:<8} Accuracy={result['accuracy']:.4f} Precision={result['fraud_precision']:.4f} "
              f"Recall={result['fraud_recall']:.4f} F1={result['fraud_f1']:.4f} PR-AUC={result['average_precision']:.4f}")


def window_indices_for_rows(rows, count, length):
    offsets = torch.arange(length - 1, -1, -1, device=rows.device)
    return (rows[:, None] - offsets[None, :]).clamp_min(0)


if __name__ == "__main__":
    main()
