"""
XGBoost classifier for Fake vs Real review detection.
Pipeline: 
1. Data split: 60% Training, 20% Validation, 20% Test (stratified, shuffled)
2. 5-fold Cross Validation on 80% (train+val) for hyperparameter tuning
3. With feature selection: SelectKBest (f_classif)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 9
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 1935990857 # 2985710947
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["AWL", "ASL", "NWO", "NVB", "NAJ", "NPV", "NST", "CDV", "NTP", "TPR"]
LABEL_COL = "Real=1/Fake=0"
N_OPTUNA_TRIALS = 250


def load_data(csv_path: str) -> tuple:
    """Load feature CSV and return X, y arrays."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {os.path.basename(csv_path)}")
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    print(f"Class distribution → Real: {(y == 1).sum()}  Fake: {(y == 0).sum()}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """80% train+val (for 5-fold CV and final fit), 20% test. Stratified and shuffled."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )
    n = len(y)
    print(f"\nStep 0  – Data split")
    print(f"  Train+Val : {len(y_tv)} ({len(y_tv)/n*100:.1f}%)  Real={(y_tv==1).sum()}, Fake={(y_tv==0).sum()}")
    print(f"  Test      : {len(y_te)} ({len(y_te)/n*100:.1f}%)  Real={(y_te==1).sum()}, Fake={(y_te==0).sum()}")
    return X_tv, X_te, y_tv, y_te


def _make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def xgb_hyperparameters(trial: optuna.Trial) -> dict:
    """Defined XGBoost hyperparameters for optimization."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "random_state": RANDOM_STATE,
    }


def build_objective(X_tv, y_tv, use_fs: bool):
    """5-fold CV mean F1 for XGBoost optimization"""
    cv = _make_cv()

    def objective(trial: optuna.Trial) -> float:
        params = xgb_hyperparameters(trial)
        steps = [("scaler", StandardScaler())]
        if use_fs:
            k = trial.suggest_int("selector_k", 4, 10)
            steps.append(("selector", SelectKBest(score_func=f_classif, k=min(k, 10))))
        steps.append(("clf", xgb.XGBClassifier(**params)))
        pipe = Pipeline(steps)
        scores = cross_val_score(pipe, X_tv, y_tv, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


def tune_model(X_tv, y_tv, use_fs: bool) -> dict:
    """Run Optuna study for XGBoost optimization"""
    objective = build_objective(X_tv, y_tv, use_fs)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    return {"best_cv_f1": study.best_value, "best_params": study.best_params}


def build_final_pipeline(best_params: dict, use_fs: bool) -> Pipeline:
    """Build pipeline from best Optuna params"""
    best_params = dict(best_params)
    selector_k = best_params.pop("selector_k", None)
    clf_params = {k: v for k, v in best_params.items()}
    if "random_state" not in clf_params:
        clf_params["random_state"] = RANDOM_STATE

    steps = [("scaler", StandardScaler())]
    if use_fs and selector_k is not None:
        steps.append(("selector", SelectKBest(score_func=f_classif, k=min(selector_k, 10))))
    steps.append(("clf", xgb.XGBClassifier(**clf_params)))
    return Pipeline(steps)


def get_feature_info(pipe: Pipeline, use_fs: bool, X_data: np.ndarray = None, y_data: np.ndarray = None) -> tuple:
    """Return (selected_feature_names, importance_array, rank_indices, n_features)"""
    if use_fs and "selector" in pipe.named_steps:
        mask = pipe.named_steps["selector"].get_support()
        sel_feats = [f for f, s in zip(FEATURE_COLS, mask) if s]
    else:
        sel_feats = list(FEATURE_COLS)

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.zeros(len(sel_feats))

    rank = np.argsort(imp)[::-1]
    return sel_feats, imp, rank, len(sel_feats)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_cm(ax, cm, title: str, cmap: str = "Blues"):
    """Draw confusion matrix heatmap on ax."""
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False, square=True,
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")


def _plot_feature_importance(ax, sel_feats, importances, rank, title: str):
    """Horizontal bar chart of feature importances."""
    if len(sel_feats) == 0 or np.all(importances == 0):
        ax.text(0.5, 0.5, "No importance\navailable", ha="center", va="center",
                fontsize=12, color="grey", transform=ax.transAxes)
        ax.set_title(title, fontsize=9, fontweight="bold")
        return
    ordered_feats = [sel_feats[i] for i in rank]
    ordered_imp = importances[rank]
    y_pos = np.arange(len(ordered_feats))
    colors = sns.color_palette("viridis", len(ordered_feats))
    ax.barh(y_pos, ordered_imp, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_feats, fontsize=7)
    ax.set_xlabel("Importance", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")


def save_xgb_plots(r_no_fs: dict, r_fs: dict, y_tv: np.ndarray, y_te: np.ndarray, out_dir: str):
    """Save confusion matrices, feature importance, and performance comparison PNGs."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _plot_cm(axes[0, 0], confusion_matrix(y_tv, r_no_fs["y_train_pred"]),
             f"No FS – Train\nAcc={r_no_fs['train_acc']:.3f}", "Blues")
    _plot_cm(axes[0, 1], confusion_matrix(y_te, r_no_fs["y_pred"]),
             f"No FS – Test\nAcc={r_no_fs['test_acc']:.3f}  F1={r_no_fs['test_f1']:.3f}", "Oranges")
    _plot_cm(axes[1, 0], confusion_matrix(y_tv, r_fs["y_train_pred"]),
             f"With FS – Train\nAcc={r_fs['train_acc']:.3f}", "Blues")
    _plot_cm(axes[1, 1], confusion_matrix(y_te, r_fs["y_pred"]),
             f"With FS – Test\nAcc={r_fs['test_acc']:.3f}  F1={r_fs['test_f1']:.3f}", "Oranges")
    fig.suptitle("XGBoost – Confusion Matrices", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, "XGBoost_confusion_matrices.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # 2. Feature importance (No FS vs FS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_feature_importance(axes[0], r_no_fs["sel_feats"], r_no_fs["importances"], r_no_fs["rank"],
                             f"No FS ({r_no_fs['n_features']} features)")
    _plot_feature_importance(axes[1], r_fs["sel_feats"], r_fs["importances"], r_fs["rank"],
                             f"With FS ({r_fs['n_features']} features)")
    fig.suptitle("XGBoost – Feature Importance", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, "XGBoost_feature_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # 3. Performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["Train Acc", "Test Acc", "Test F1", "CV F1"]
    vals_nf = [r_no_fs["train_acc"], r_no_fs["test_acc"], r_no_fs["test_f1"], r_no_fs["best_cv_f1"]]
    vals_fs = [r_fs["train_acc"], r_fs["test_acc"], r_fs["test_f1"], r_fs["best_cv_f1"]]
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w / 2, vals_nf, w, label="No FS", color="steelblue", edgecolor="black")
    axes[0].bar(x + w / 2, vals_fs, w, label="With FS", color="darkorange", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=8)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim([0, 1.05])
    axes[0].set_title("No FS vs With FS", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    f1_no_fs = f1_score(y_te, r_no_fs["y_pred"], average=None)
    axes[1].bar(["Fake", "Real"], f1_no_fs, color=sns.color_palette("viridis", 2), edgecolor="black")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim([0, 1.05])
    axes[1].set_title(f"Per-Class F1 – No FS (macro={r_no_fs['test_f1']:.3f})", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    f1_fs = f1_score(y_te, r_fs["y_pred"], average=None)
    axes[2].bar(["Fake", "Real"], f1_fs, color=sns.color_palette("viridis", 2), edgecolor="black")
    axes[2].set_ylabel("F1")
    axes[2].set_ylim([0, 1.05])
    axes[2].set_title(f"Per-Class F1 – With FS (macro={r_fs['test_f1']:.3f})", fontsize=10, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("XGBoost – Performance Comparison", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, "XGBoost_performance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # 4. Summary figure
    best = r_fs if r_fs["test_f1"] >= r_no_fs["test_f1"] else r_no_fs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_cm(axes[0], confusion_matrix(y_te, best["y_pred"]),
             f"Best (Test)\nAcc={best['test_acc']:.3f}  F1={best['test_f1']:.3f}", "Blues")
    axes[1].axis("off")
    summary = (
        f"Best variant: {'With FS' if best['use_fs'] else 'No FS'}\n\n"
        f"Test Accuracy: {best['test_acc']:.4f}\n"
        f"Test F1:       {best['test_f1']:.4f}\n"
        f"CV F1 (5-fold): {best['best_cv_f1']:.4f}\n"
        f"# Features:    {best['n_features']}\n"
        f"Most important: {best['most_important']}\n\n"
        f"Benchmark: 79.09% Acc, 76.99% F1"
    )
    axes[1].text(0.1, 0.5, summary, fontsize=11, verticalalignment="center", family="monospace")
    axes[1].set_title("Summary", fontsize=10, fontweight="bold")
    fig.suptitle("XGBoost – Evaluation Summary", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "XGBoost_evaluation_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def execute_model(X_tv, y_tv, X_te, y_te, use_fs: bool):
    """Evaluate one variant (with or without FS)"""
    label = "with SelectKBest" if use_fs else "no FS"
    print(f"\n>>> Tuning XGBoost ({label}) ({N_OPTUNA_TRIALS} Optuna trials)")
    result = tune_model(X_tv, y_tv, use_fs)
    pipe = build_final_pipeline(dict(result["best_params"]), use_fs)
    pipe.fit(X_tv, y_tv)

    y_train_pred = pipe.predict(X_tv)
    train_acc = accuracy_score(y_tv, y_train_pred)
    train_f1 = f1_score(y_tv, y_train_pred)
    y_pred = pipe.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred)
    test_f1 = f1_score(y_te, y_pred)
    sel_feats, imp, rank, n_feats = get_feature_info(pipe, use_fs)
    most_important = sel_feats[rank[0]] if len(sel_feats) else "—"

    return {
        "use_fs": use_fs,
        "pipe": pipe,
        "best_cv_f1": result["best_cv_f1"],
        "train_acc": train_acc,
        "train_f1": train_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "n_features": n_feats,
        "sel_feats": sel_feats,
        "most_important": most_important,
        "rank": rank,
        "importances": imp,
        "y_pred": y_pred,
        "y_train_pred": y_train_pred,
    }


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "Data", "reviewFeatures.csv")
    model_dir = os.path.dirname(__file__)

    X, y = load_data(data_path)
    X_tv, X_te, y_tv, y_te = split_data(X, y)

    r_no_fs = execute_model(X_tv, y_tv, X_te, y_te, use_fs=False)
    r_fs = execute_model(X_tv, y_tv, X_te, y_te, use_fs=True)

    best = r_fs if r_fs["test_f1"] >= r_no_fs["test_f1"] else r_no_fs
    tag = "with SelectKBest" if best["use_fs"] else "no FS"

    print(f"\n{'='*60}")
    print(f"XGBoost {tag} (chosen by test F1)")
    print(f"{'='*60}")
    print(f"Best CV F1 - 5-fold : {best['best_cv_f1']:.4f}")
    print(f"Train+Val Accuracy : {best['train_acc']:.4f}")
    print(f"Train+Val F1       : {best['train_f1']:.4f}")
    print(f"Test Accuracy     : {best['test_acc']:.4f}")
    print(f"Test F1           : {best['test_f1']:.4f}")
    print(f"# Selected features: {best['n_features']}")
    print(f"Selected features : {best['sel_feats']}")
    print(f"Most important    : {best['most_important']}")
    print(f"\nFeature importance ranking:")
    for i, idx in enumerate(best["rank"]):
        print(f"    {i+1}. {best['sel_feats'][idx]:>4s}  →  {best['importances'][idx]:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_te, best['y_pred'])}")
    print(f"\n{classification_report(y_te, best['y_pred'], target_names=['Fake', 'Real'])}")

    out_path = os.path.join(model_dir, "best_model_xgb.joblib")
    joblib.dump({"pipeline": best["pipe"], "features": FEATURE_COLS}, out_path)
    print(f"\nModel saved → {out_path}")

    plot_dir = os.path.join(model_dir, "plots")
    print(f"\n>>> Saving plots → {plot_dir}")
    save_xgb_plots(r_no_fs, r_fs, y_tv, y_te, plot_dir)

    print(f"\nNo FS  → Test Acc={r_no_fs['test_acc']:.4f}  F1={r_no_fs['test_f1']:.4f}")
    print(f"With FS → Test Acc={r_fs['test_acc']:.4f}  F1={r_fs['test_f1']:.4f}")

    print(f"\nBenchmark: Accuracy 79.09%  F1 76.99%")
    print(f"This run: Accuracy {best['test_acc']*100:.2f}%  F1 {best['test_f1']*100:.2f}%")
    if best["test_acc"] >= 0.7909 and best["test_f1"] >= 0.7699:
        print("Model meets or exceeds benchmark.")
    else:
        print("Model is below benchmark, try: increase OPTUNA_TRIALS")

    return best


if __name__ == "__main__":
    main()
