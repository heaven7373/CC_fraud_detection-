"""CardGuard: full-split credit-card fraud model comparison dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODEL_COLORS = {
    "ANN": "#2256A3",
    "LSTM": "#27AE60",
}
MODEL_ICONS = {"ANN": "🧠", "LSTM": "🔁"}
PROBA_COLS = {"ANN": "ann_probability", "LSTM": "lstm_probability"}

st.set_page_config(page_title="CardGuard | Fraud Detection", page_icon="💳", layout="wide")


@st.cache_data
def load_split(filename):
    path = DATA / filename
    return pd.read_csv(path, parse_dates=["trans_date_trans_time"]) if path.exists() else None


@st.cache_data
def load_scores(modified_time=None):
    path = MODELS / "fraud_metrics.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw and "model" in raw:
        return {"ANN": raw}
    return raw


@st.cache_data
def load_predictions(modified_time=None):
    path = MODELS / "fraud_test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


train = load_split("fraud_train.csv")
test = load_split("fraud_test.csv")
metrics_path = MODELS / "fraud_metrics.json"
predictions_path = MODELS / "fraud_test_predictions.csv"
all_scores = load_scores(metrics_path.stat().st_mtime_ns if metrics_path.exists() else None) or {}
predictions = load_predictions(predictions_path.stat().st_mtime_ns if predictions_path.exists() else None)
available = [name for name in MODEL_COLORS if name in all_scores]


def overview():
    st.title("Credit Card Fraud Detection")
    st.caption("Train on the complete training split and evaluate on the complete held-out test split.")
    if train is None or test is None:
        st.error("Prepared train and test files are missing. Run the dataset preparation script first.")
        return
    train_fraud, test_fraud = int(train.is_fraud.sum()), int(test.is_fraud.sum())
    cards = st.columns(4)
    cards[0].metric("Training transactions", f"{len(train):,}")
    cards[1].metric("Held-out test transactions", f"{len(test):,}")
    cards[2].metric("Training fraud cases", f"{train_fraud:,}", delta=f"{train_fraud / len(train):.2%} of train", delta_color="off")
    cards[3].metric("Test fraud cases", f"{test_fraud:,}", delta=f"{test_fraud / len(test):.2%} of test", delta_color="off")
    st.subheader("Project overview")
    st.write("ANN and LSTM are trained on every row of fraudTrain.csv and evaluated on every row of fraudTest.csv. Because fraud is rare, compare precision and recall alongside accuracy.")
    left, right = st.columns(2)
    with left:
        balance = pd.DataFrame({"Split": ["Train", "Train", "Test", "Test"], "Class": ["Legitimate", "Fraud", "Legitimate", "Fraud"], "Transactions": [int((train.is_fraud == 0).sum()), train_fraud, int((test.is_fraud == 0).sum()), test_fraud]})
        st.plotly_chart(px.bar(balance, x="Split", y="Transactions", color="Class", barmode="group", title="Train and test class balance", color_discrete_map={"Legitimate": "#2256A3", "Fraud": "#C43D4B"}), width="stretch")
    with right:
        combined = pd.concat([train.assign(Split="Train"), test.assign(Split="Test")], ignore_index=True)
        by_category = combined.groupby(["category", "Split", "is_fraud"], as_index=False).size()
        by_category["Class"] = by_category.is_fraud.map({0: "Legitimate", 1: "Fraud"})
        st.plotly_chart(px.bar(by_category, x="category", y="size", color="Class", facet_col="Split", barmode="group", title="Transactions by category", color_discrete_map={"Legitimate": "#2256A3", "Fraud": "#C43D4B"}), width="stretch")
    st.info("Both models use the same train/test split. The prepared files contain no card/customer ID, so LSTM uses chronological transaction windows rather than per-card histories.")


def model_comparison():
    st.title("Model Comparison")
    st.caption("Every training and test row is included. Training scores show fit; held-out test scores show generalization.")
    if not available:
        st.warning("No model comparison results yet. Run `python scripts\\train_fraud_model.py`.")
        return
    train_metric_columns = {"Accuracy": "accuracy", "Fraud precision": "fraud_precision", "Fraud recall": "fraud_recall", "Fraud F1": "fraud_f1"}
    test_metric_columns = {**train_metric_columns, "ROC–AUC": "roc_auc", "PR–AUC": "average_precision"}
    train_rows, test_rows = [], []
    for name in available:
        scores = all_scores[name]
        train_values = scores.get("train_metrics")
        if train_values:
            train_rows.append({"Model": f"{MODEL_ICONS[name]} {name}", **{label: train_values[key] for label, key in train_metric_columns.items()}})
        test_rows.append({"Model": f"{MODEL_ICONS[name]} {name}", **{label: scores[key] for label, key in test_metric_columns.items()}, "Training time (s)": scores.get("train_time_sec", 0)})
    train_table = pd.DataFrame(train_rows).set_index("Model") if train_rows else None
    test_table = pd.DataFrame(test_rows).set_index("Model")
    train_tab, test_tab = st.tabs(["Full training split", "Full held-out test split"])
    with train_tab:
        if train_table is None:
            st.info("Training metrics are not present yet. Retrain models with `python scripts\\train_fraud_model.py`.")
        else:
            st.caption(f"Metrics use all {len(train):,} training rows at the same 0.50 decision threshold.")
            st.dataframe(train_table.style.format("{:.3f}").highlight_max(axis=0, color="#dff2e1"), width="stretch")
    with test_tab:
        st.caption(f"Metrics use all {len(test):,} held-out test rows at the same 0.50 decision threshold.")
        st.dataframe(test_table.style.format("{:.3f}").highlight_max(axis=0, color="#dff2e1"), width="stretch")
    st.caption("Compare training and test scores to spot overfitting. Fraud precision, recall, F1, and PR–AUC are more informative than accuracy alone for this imbalanced dataset.")
    split_choice = st.radio("Chart split", ["Held-out test", "Training"], horizontal=True)
    metric_columns = test_metric_columns if split_choice == "Held-out test" else train_metric_columns
    selected = st.multiselect("Metrics to compare", list(metric_columns), default=["Accuracy", "Fraud precision", "Fraud recall", "Fraud F1"], key="comparison_metrics")
    if selected:
        chart_rows = []
        for name in available:
            values = all_scores[name] if split_choice == "Held-out test" else all_scores[name].get("train_metrics", {})
            chart_rows.extend({"Model": name, "Metric": label, "Score": values[metric_columns[label]]} for label in selected if metric_columns[label] in values)
        fig = px.bar(pd.DataFrame(chart_rows), x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f", color_discrete_map=MODEL_COLORS, title="Metric comparison")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    with left:
        roc_fig = go.Figure()
        roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line={"dash": "dash", "color": "gray"})
        for name in available:
            curve = all_scores[name]["roc_curve"]
            roc_fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{name} · {all_scores[name]['roc_auc']:.3f}", line={"color": MODEL_COLORS[name]}))
        roc_fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(roc_fig, width="stretch")
    with right:
        pr_fig = go.Figure()
        for name in available:
            curve = all_scores[name]["precision_recall_curve"]
            pr_fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=f"{name} · {all_scores[name]['average_precision']:.3f}", line={"color": MODEL_COLORS[name]}))
        pr_fig.update_layout(title="Precision–recall curves", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(pr_fig, width="stretch")


def ann_model_lab():
    st.title("ANN Model Lab")
    st.caption("How the neural network learns from the complete training split.")
    if train is None or test is None:
        st.error("Prepared train and test files are missing.")
        return
    ann = all_scores.get("ANN", {})
    cards = st.columns(4)
    cards[0].metric("Training rows", f"{len(train):,}")
    cards[1].metric("Fraud rows", f"{int(train.is_fraud.sum()):,}")
    cards[2].metric("Legitimate rows", f"{int((train.is_fraud == 0).sum()):,}")
    cards[3].metric("Test rows held out", f"{len(test):,}")
    st.markdown("**Architecture:** 128 → 64 → 32 hidden neurons · ReLU · Adam · early stopping")
    st.markdown("**Inputs:** amount, hour, day, weekday, city population, category, gender, and job")
    st.markdown("**Imbalance handling:** class weights; every training row is retained")
    if ann.get("training_history"):
        history = ann["training_history"]
        frame = pd.DataFrame({"Epoch": range(1, len(history["loss"]) + 1), "Training loss": history["loss"], "Validation accuracy": history["validation_accuracy"]})
        st.plotly_chart(px.line(frame, x="Epoch", y=["Training loss", "Validation accuracy"], markers=True, title="ANN training history"), width="stretch")
        st.caption(f"Early stopping completed training after {ann['epochs']} epochs.")
    else:
        st.info("Run the training script to create the ANN history and comparison results.")
    lstm = all_scores.get("LSTM", {})
    st.divider()
    st.subheader("LSTM sequence model")
    st.write(f"The LSTM reads each transaction together with the previous {lstm.get('sequence_length', 10) - 1} chronological transactions in that split. It uses the same full training and held-out test files as ANN.")
    st.caption("The dataset supplied to this app omits a card/customer identifier. These are chronological market-wide windows, not individual card histories.")
    if lstm.get("training_history"):
        history = lstm["training_history"]
        st.plotly_chart(px.line(pd.DataFrame({"Epoch": range(1, len(history["loss"]) + 1), "Training loss": history["loss"]}), x="Epoch", y="Training loss", markers=True, title="LSTM training loss"), width="stretch")
        st.caption(f"Training ran on {lstm.get('device', 'CPU')}.")


def data_explorer():
    st.title("Data Explorer")
    st.caption("Filter and inspect the prepared transaction records.")
    if train is None or test is None:
        st.error("Prepared train and test files are missing.")
        return
    split = st.selectbox("Dataset split", ["Train", "Test"])
    frame = train if split == "Train" else test
    controls = st.columns(3)
    category = controls[0].selectbox("Category", ["All categories", *sorted(frame.category.dropna().unique())])
    transaction_class = controls[1].selectbox("Class", ["All classes", "Legitimate", "Fraud"])
    row_limit = controls[2].selectbox("Rows to display", [25, 50, 100, 250], index=1)
    filtered = frame
    if category != "All categories":
        filtered = filtered[filtered.category == category]
    if transaction_class != "All classes":
        filtered = filtered[filtered.is_fraud == (1 if transaction_class == "Fraud" else 0)]
    st.caption(f"{len(frame):,} rows in {split.lower()} · {len(filtered):,} match the filters. The table is a preview; all rows were used for model training/evaluation.")
    st.dataframe(filtered.head(row_limit), width="stretch", hide_index=True)


def final_report():
    st.title("Final Report")
    st.caption("Held-out test results and fraud-pattern graphs.")
    if not available or predictions is None or test is None:
        st.warning("Run `python scripts\\train_fraud_model.py` to generate the full model comparison and report.")
        return
    selected_model = st.selectbox("Model", available)
    scores = all_scores[selected_model]
    probability_column = PROBA_COLS[selected_model]
    predictions["Actual class"] = predictions.actual_fraud.map({0: "Legitimate", 1: "Fraud"})
    cards = st.columns(5)
    cards[0].metric("Test rows", f"{scores['test_rows']:,}")
    cards[1].metric("Accuracy", f"{scores['accuracy']:.1%}")
    cards[2].metric("ROC–AUC", f"{scores['roc_auc']:.3f}")
    cards[3].metric("PR–AUC", f"{scores['average_precision']:.3f}")
    cards[4].metric("Fraud precision", f"{scores['fraud_precision']:.1%}")
    cards = st.columns(3)
    cards[0].metric("Fraud recall", f"{scores['fraud_recall']:.1%}")
    cards[1].metric("Fraud F1", f"{scores['fraud_f1']:.3f}")
    cards[2].metric("Training time", f"{scores.get('train_time_sec', 0):.1f}s")
    left, right = st.columns(2)
    with left:
        matrix = scores["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"], text=matrix, texttemplate="%{text}", colorscale="Blues"))
        fig.update_layout(title=f"Confusion matrix · {selected_model}")
        st.plotly_chart(fig, width="stretch")
    with right:
        roc = scores["roc_curve"]
        st.plotly_chart(px.line(pd.DataFrame({"FPR": roc["fpr"], "TPR": roc["tpr"]}), x="FPR", y="TPR", title="ROC curve"), width="stretch")
    left, right = st.columns(2)
    with left:
        pr = scores["precision_recall_curve"]
        st.plotly_chart(px.line(pd.DataFrame({"Recall": pr["recall"], "Precision": pr["precision"]}), x="Recall", y="Precision", title="Precision–recall curve"), width="stretch")
    with right:
        st.plotly_chart(px.histogram(predictions, x=probability_column, color="Actual class", nbins=45, barmode="overlay", opacity=0.7, title="Fraud score distribution"), width="stretch")
    st.subheader("Fraud patterns · full test split")
    left, right = st.columns(2)
    with left:
        category_stats = test.groupby("category", as_index=False).agg(transactions=("is_fraud", "size"), fraud_cases=("is_fraud", "sum"))
        category_stats["Fraud rate (%)"] = 100 * category_stats.fraud_cases / category_stats.transactions
        st.plotly_chart(px.bar(category_stats.sort_values("Fraud rate (%)", ascending=False), x="category", y="Fraud rate (%)", title="Fraud rate by category"), width="stretch")
    with right:
        daily = test.assign(Date=test.trans_date_trans_time.dt.date).groupby("Date", as_index=False).agg(fraud_cases=("is_fraud", "sum"))
        st.plotly_chart(px.line(daily, x="Date", y="fraud_cases", title="Fraud over time"), width="stretch")
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(scores["classification_report"]).T, width="stretch")
    st.warning("These results use a simulated educational dataset and do not establish performance on real banking transactions.")


with st.sidebar:
    st.title("💳 CardGuard", anchor=False)
    st.caption("CREDIT CARD FRAUD DETECTION")
    st.divider()
    if available:
        best = max(available, key=lambda model: all_scores[model]["fraud_f1"])
        st.markdown("**Model scorecard**")
        for model in available:
            score = all_scores[model]
            st.caption(f"{MODEL_ICONS[model]} {model} · Precision {score['fraud_precision']:.3f} · Recall {score['fraud_recall']:.3f}")
        st.caption(f"Best F1: {best}")

pages = [
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(model_comparison, title="Model Comparison", icon=":material/compare_arrows:"),
    st.Page(ann_model_lab, title="ANN & LSTM Lab", icon=":material/psychology:"),
    st.Page(data_explorer, title="Data Explorer", icon=":material/table_view:"),
    st.Page(final_report, title="Final Report", icon=":material/description:"),
]
st.navigation(pages, position="sidebar", expanded=True).run()
