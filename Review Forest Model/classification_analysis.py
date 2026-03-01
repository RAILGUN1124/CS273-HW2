"""
Restaurant Review Classifier – Fake vs Real
=============================================
Pipeline:
  Step 0: Hold out 20% stratified test set
  Step 1: 5-fold Stratified CV on remaining 80% (train+val) for hyper-param tuning
  Step 2: Select best hyperparameters (by mean CV F1) via Optuna
  Step 3: Retrain on full train+val with best params
  Step 4: Evaluate once on held-out test set

Two variants compared:
  A) No feature selection  (all 10 features)
  B) With feature selection (SelectKBest + f_classif)

Models tried: Random Forest, Gradient Boosting, SVM
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 9

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 1935990857
# RANDOM_STATE = 25105967
np.random.seed(RANDOM_STATE)

# ── Constants ──────────────────────────────────────────────────────────────
FEATURE_COLS = ["AWL", "ASL", "NWO", "NVB", "NAJ", "NPV", "NST", "CDV", "NTP", "TPR"]
LABEL_COL = "Real=1/Fake=0"
N_OPTUNA_TRIALS = 300


# =========================================================================
#  Data loading & splitting
# =========================================================================

def load_data(csv_path: str) -> tuple:
    """Load feature CSV and return X, y arrays."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {os.path.basename(csv_path)}")
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    print(f"Class distribution  → Real: {(y == 1).sum()}  Fake: {(y == 0).sum()}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """Step 0: 80/20 stratified split into (train+val) and test."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y, shuffle=True,
    )
    print(f"\nStep 0  – Data split")
    print(f"  Train+Val : {len(y_tv)}  (Real={(y_tv==1).sum()}, Fake={(y_tv==0).sum()})")
    print(f"  Test      : {len(y_te)}  (Real={(y_te==1).sum()}, Fake={(y_te==0).sum()})")
    return X_tv, X_te, y_tv, y_te


# =========================================================================
#  Optuna objective factories (one per model type)
# =========================================================================

def _make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def print_cv_stratification(X, y):
    """Print per-fold class distribution to verify stratification."""
    cv = _make_cv()
    print(f"\nStep 1  – 5-Fold Stratified CV  (verifying stratification)")
    print(f"  {'Fold':<6} {'Train Real':>10} {'Train Fake':>10} {'Train %Real':>11}  │  {'Val Real':>8} {'Val Fake':>8} {'Val %Real':>9}")
    print(f"  {'-'*72}")
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        y_tr, y_val = y[train_idx], y[val_idx]
        tr_real, tr_fake = (y_tr == 1).sum(), (y_tr == 0).sum()
        va_real, va_fake = (y_val == 1).sum(), (y_val == 0).sum()
        print(f"  {i:<6} {tr_real:>10} {tr_fake:>10} {tr_real/len(y_tr)*100:>10.1f}%  │  {va_real:>8} {va_fake:>8} {va_real/len(y_val)*100:>8.1f}%")
    total_real_pct = (y == 1).sum() / len(y) * 100
    print(f"  Overall class ratio: {total_real_pct:.1f}% Real")


def _suggest_rf(trial: optuna.Trial) -> dict:
    """Suggest Random Forest hyperparameters."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10, 15, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE,
    }


MODEL_REGISTRY = {
    "RandomForest":      (RandomForestClassifier, _suggest_rf),
}


def _build_objective(model_cls, suggest_fn, X_tv, y_tv, use_fs: bool):
    """Return an Optuna objective that does 5-fold CV and returns mean F1."""
    cv = _make_cv()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)

        steps = [("scaler", StandardScaler())]
        if use_fs:
            k = trial.suggest_int("selector_k", 3, 9) #Only half of the features, at least 3 to avoid too small feature sets and at most 9 to not be the same as baseline (all features)
            score_func_name = trial.suggest_categorical("score_func", ["f_classif", "mutual_info"])
            score_func = f_classif if score_func_name == "f_classif" else mutual_info_classif
            steps.append(("selector", SelectKBest(score_func=score_func, k=k)))
        steps.append(("clf", model_cls(**params)))

        pipe = Pipeline(steps)
        scores = cross_val_score(pipe, X_tv, y_tv, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


# =========================================================================
#  Hyperparameter search + final training
# =========================================================================

def tune_model(model_name: str, X_tv, y_tv, use_fs: bool) -> dict:
    """Run Optuna study for a single (model, feature-selection) variant."""
    model_cls, suggest_fn = MODEL_REGISTRY[model_name]
    objective = _build_objective(model_cls, suggest_fn, X_tv, y_tv, use_fs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    return {
        "best_cv_f1": study.best_value,
        "best_params": study.best_params,
    }


def build_final_pipeline(model_name: str, best_params: dict, use_fs: bool) -> Pipeline:
    """Construct a Pipeline from the best Optuna params and return it (unfitted)."""
    model_cls, _ = MODEL_REGISTRY[model_name]

    # Separate selector params from model params
    selector_k = best_params.pop("selector_k", None)
    score_func_name = best_params.pop("score_func", "f_classif")
    clf_params = {k: v for k, v in best_params.items()}

    # Ensure constant keys present for model constructors
    if model_name not in ("KNN", "LDA") and "random_state" not in clf_params:
        clf_params["random_state"] = RANDOM_STATE
    if model_name == "SVM":
        clf_params["probability"] = True
    if model_name == "LogisticRegression":
        clf_params["solver"] = "saga"
        clf_params["max_iter"] = 5000
    if model_name == "LDA":
        # Translate Optuna's synthetic shrinkage params
        shrinkage_type = clf_params.pop("shrinkage_type", "none")
        shrinkage_val = clf_params.pop("shrinkage_val", None)
        if shrinkage_type == "auto":
            clf_params["shrinkage"] = "auto"
        elif shrinkage_type == "float" and shrinkage_val is not None:
            clf_params["shrinkage"] = shrinkage_val

    steps = [("scaler", StandardScaler())]
    if use_fs and selector_k is not None:
        score_func = f_classif if score_func_name == "f_classif" else mutual_info_classif
        steps.append(("selector", SelectKBest(score_func=score_func, k=selector_k)))
    steps.append(("clf", model_cls(**clf_params)))

    return Pipeline(steps)


def evaluate_on_test(pipeline: Pipeline, X_test, y_test) -> tuple:
    """Return (accuracy, f1, y_pred)."""
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), y_pred


def get_feature_info(pipeline: Pipeline, use_fs: bool,
                     X_data: np.ndarray = None, y_data: np.ndarray = None) -> tuple:
    """Return (selected_feature_names, importance_array, ranked_indices).

    Uses native feature_importances_ or coef_ when available;
    falls back to permutation importance on the supplied data.
    """
    if use_fs and "selector" in pipeline.named_steps:
        mask = pipeline.named_steps["selector"].get_support()
        sel_feats = [f for f, s in zip(FEATURE_COLS, mask) if s]
    else:
        sel_feats = list(FEATURE_COLS)

    # True number of features the classifier actually sees
    actual_n_features = len(sel_feats)

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
        if len(imp) != len(sel_feats):
            imp = np.mean(np.abs(clf.coef_), axis=0)
    elif X_data is not None and y_data is not None:
        # Permutation importance fallback (KNN, SVM-rbf, etc.)
        # Computed on the full pipeline, so importance is per *input* feature
        perm = permutation_importance(
            pipeline, X_data, y_data,
            n_repeats=10, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1,
        )
        imp = perm.importances_mean
        # Override sel_feats to match full input (pipeline handles selection internally)
        sel_feats = list(FEATURE_COLS)
    else:
        imp = np.zeros(len(sel_feats))

    rank = np.argsort(imp)[::-1]
    return sel_feats, imp, rank, actual_n_features


# =========================================================================
#  Visualization helpers
# =========================================================================

def _plot_confusion_matrix(ax, cm, title, cmap="Blues"):
    """Draw a single confusion matrix heatmap on ax."""
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False, square=True,
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")


def _plot_feature_importance(ax, sel_feats, importances, rank, title):
    """Horizontal bar chart of feature importances."""
    if np.all(importances == 0):
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


def save_model_plots(model_name: str, r_no_fs: dict, r_fs: dict,
                     y_tv: np.ndarray, y_te: np.ndarray, out_dir: str):
    """Generate 3 PNGs for a single model (no-FS & with-FS variants).

    Files produced:
      {model}_confusion_matrices.png
      {model}_feature_importance.png
      {model}_performance.png
    """
    # ── PNG 1: Confusion Matrices (2×2) ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    train_cm_nf = confusion_matrix(y_tv, r_no_fs["y_train_pred"])
    test_cm_nf  = confusion_matrix(y_te, r_no_fs["y_pred"])
    train_cm_fs = confusion_matrix(y_tv, r_fs["y_train_pred"])
    test_cm_fs  = confusion_matrix(y_te, r_fs["y_pred"])

    _plot_confusion_matrix(axes[0, 0], train_cm_nf,
        f"No FS – Train\nAcc={r_no_fs['train_acc']:.3f}", "Blues")
    _plot_confusion_matrix(axes[0, 1], test_cm_nf,
        f"No FS – Test\nAcc={r_no_fs['acc']:.3f}  F1={r_no_fs['f1']:.3f}", "Oranges")
    _plot_confusion_matrix(axes[1, 0], train_cm_fs,
        f"With FS – Train\nAcc={r_fs['train_acc']:.3f}", "Blues")
    _plot_confusion_matrix(axes[1, 1], test_cm_fs,
        f"With FS – Test\nAcc={r_fs['acc']:.3f}  F1={r_fs['f1']:.3f}", "Oranges")

    fig.suptitle(f"{model_name} – Confusion Matrices", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_confusion_matrices.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # ── PNG 2: Feature Importance ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_feature_importance(axes[0], r_no_fs["sel_feats"], r_no_fs["importances"],
                             r_no_fs["rank"],
                             f"No FS ({r_no_fs['n_features']} features)")
    _plot_feature_importance(axes[1], r_fs["sel_feats"], r_fs["importances"],
                             r_fs["rank"],
                             f"With FS ({r_fs['n_features']} features)")
    fig.suptitle(f"{model_name} – Feature Importance", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_feature_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # ── PNG 3: Performance Comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a – Metric bars: No FS vs FS
    metrics = ["Train Acc", "Test Acc", "Test F1", "CV F1"]
    vals_nf = [r_no_fs["train_acc"], r_no_fs["acc"], r_no_fs["f1"], r_no_fs["best_cv_f1"]]
    vals_fs = [r_fs["train_acc"], r_fs["acc"], r_fs["f1"], r_fs["best_cv_f1"]]
    x = np.arange(len(metrics)); w = 0.35
    axes[0].bar(x - w/2, vals_nf, w, label="No FS", color="steelblue", edgecolor="black")
    axes[0].bar(x + w/2, vals_fs, w, label="With FS", color="darkorange", edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, fontsize=8)
    axes[0].set_ylabel("Score"); axes[0].set_ylim([0, 1.05])
    axes[0].set_title("No FS vs With FS", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis="y")

    # 3b – Per-class F1 (No FS)
    f1_pc_nf = f1_score(y_te, r_no_fs["y_pred"], average=None)
    classes = ["Fake", "Real"]
    colors_c = sns.color_palette("viridis", 2)
    axes[1].bar(classes, f1_pc_nf, color=colors_c, edgecolor="black")
    axes[1].set_ylabel("F1"); axes[1].set_ylim([0, 1.05])
    axes[1].set_title(f"Per-Class F1 – No FS (macro={r_no_fs['f1']:.3f})",
                      fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3c – Per-class F1 (FS)
    f1_pc_fs = f1_score(y_te, r_fs["y_pred"], average=None)
    axes[2].bar(classes, f1_pc_fs, color=colors_c, edgecolor="black")
    axes[2].set_ylabel("F1"); axes[2].set_ylim([0, 1.05])
    axes[2].set_title(f"Per-Class F1 – With FS (macro={r_fs['f1']:.3f})",
                      fontsize=10, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{model_name} – Performance Comparison", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_performance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

# =========================================================================
#  Reporting helpers
# =========================================================================

def print_variant_result(r: dict, y_test) -> None:
    """Print detailed results for one variant."""
    fs_tag = "With FS" if r["fs"] else "No FS"
    print(f"\n{'=' * 65}")
    print(f"  {r['name']} – {fs_tag}")
    print(f"{'=' * 65}")
    print(f"  Best CV F1 (mean) : {r['best_cv_f1']:.4f}")
    print(f"  Best Optuna params: {r['best_params']}")
    if r["fs"]:
        print(f"  Selected features : {r['sel_feats']}")
    print(f"\n  ── Train-set results ──")
    print(f"  Accuracy : {r['train_acc']:.4f}")
    print(f"\n  ── Test-set results ──")
    print(f"  Accuracy : {r['acc']:.4f}")
    print(f"  F1-score : {r['f1']:.4f}")
    print(f"  Features : {r['n_features']}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, r['y_pred'])}")
    print(f"\n{classification_report(y_test, r['y_pred'], target_names=['Fake', 'Real'])}")
    print("  Feature importance ranking:")
    for i, idx in enumerate(r["rank"]):
        print(f"    {i+1}. {r['sel_feats'][idx]:>4s}  →  {r['importances'][idx]:.4f}")


def print_summary(best_no_fs: dict, best_with_fs: dict) -> None:
    """Print the final comparison table."""
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'=' * 65}")
    print(f"{'Metric':<28} {'No FS':>14} {'With FS':>14}")
    print("-" * 58)
    print(f"{'Model':<28} {best_no_fs['name']:>14} {best_with_fs['name']:>14}")
    print(f"{'Train Accuracy':<28} {best_no_fs['train_acc']:>14.4f} {best_with_fs['train_acc']:>14.4f}")
    print(f"{'Test Accuracy':<28} {best_no_fs['acc']:>14.4f} {best_with_fs['acc']:>14.4f}")
    print(f"{'F1-score (main metric)':<28} {best_no_fs['f1']:>14.4f} {best_with_fs['f1']:>14.4f}")
    print(f"{'# Selected features':<28} {best_no_fs['n_features']:>14} {best_with_fs['n_features']:>14}")
    a, b = best_no_fs, best_with_fs
    print(f"{'Most important feature':<28} {a['sel_feats'][a['rank'][0]]:>14} "
          f"{b['sel_feats'][b['rank'][0]]:>14}")
    print("-" * 58)

    overall = best_with_fs if best_with_fs["f1"] >= best_no_fs["f1"] else best_no_fs
    print(f"\n  → Overall winner (by F1): {overall['name']} "
          f"({'with' if overall['fs'] else 'no'} feature selection)  "
          f"F1 = {overall['f1']:.4f}\n")


# =========================================================================
#  Main
# =========================================================================

def main():
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "Data", "reviewFeatures.csv")
    model_dir = os.path.dirname(__file__)

    # Load & split
    X, y = load_data(data_path)
    X_tv, X_te, y_tv, y_te = split_data(X, y)

    # Verify CV stratification on train+val set
    print_cv_stratification(X_tv, y_tv)

    # Run all (model × feature-selection) variants
    results = []
    for model_name in MODEL_REGISTRY:
        for use_fs in [False, True]:
            tag = f"{model_name} ({'FS' if use_fs else 'no FS'})"
            print(f"\n>>> Tuning: {tag}  ({N_OPTUNA_TRIALS} Optuna trials) …", flush=True)

            study_result = tune_model(model_name, X_tv, y_tv, use_fs)

            # Step 3: retrain best model on ALL train+val
            pipe = build_final_pipeline(model_name, dict(study_result["best_params"]), use_fs)
            pipe.fit(X_tv, y_tv)

            # Step 4: evaluate on train+val and test
            y_train_pred = pipe.predict(X_tv)
            train_acc = accuracy_score(y_tv, y_train_pred)
            acc, f1, y_pred = evaluate_on_test(pipe, X_te, y_te)
            sel_feats, imp, rank, actual_n_feats = get_feature_info(pipe, use_fs, X_tv, y_tv)

            r = {
                "name": model_name,
                "fs": use_fs,
                "best_cv_f1": study_result["best_cv_f1"],
                "best_params": study_result["best_params"],
                "train_acc": train_acc,
                "acc": acc,
                "f1": f1,
                "n_features": actual_n_feats,
                "sel_feats": sel_feats,
                "importances": imp,
                "rank": rank,
                "model": pipe,
                "y_pred": y_pred,
                "y_train_pred": y_train_pred,
            }
            results.append(r)

            lbl = "With FS" if use_fs else "No FS"
            print(f"    [{lbl}]  CV-F1={r['best_cv_f1']:.4f}  "
                  f"Train-Acc={train_acc:.4f}  Test-Acc={acc:.4f}  Test-F1={f1:.4f}  feats={len(sel_feats)}")


    # Pick best per category
    best_no_fs = sorted([r for r in results if not r["fs"]],
                        key=lambda r: (-r["f1"], r["n_features"]))[0]
    best_with_fs = sorted([r for r in results if r["fs"]],
                          key=lambda r: (-r["f1"], r["n_features"]))[0]

    # Detailed reports
    print_variant_result(best_no_fs, y_te)
    print_variant_result(best_with_fs, y_te)

    # Save models
    path_a = os.path.join(model_dir, "best_model_no_fs.joblib")
    joblib.dump({"pipeline": best_no_fs["model"], "features": FEATURE_COLS}, path_a)
    print(f"\n  Model saved → {path_a}")

    path_b = os.path.join(model_dir, "best_model_with_fs.joblib")
    joblib.dump({"pipeline": best_with_fs["model"], "features": FEATURE_COLS}, path_b)
    print(f"  Model saved → {path_b}")

    # Summary table
    print_summary(best_no_fs, best_with_fs)

    # ── Generate plots ─────────────────────────────────────────────────
    plot_dir = os.path.join(model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n>>> Generating plots → {plot_dir}")

    # 3 PNGs per model (no-FS and FS variants side-by-side)
    for model_name in MODEL_REGISTRY:
        r_nf = next(r for r in results if r["name"] == model_name and not r["fs"])
        r_fs = next(r for r in results if r["name"] == model_name and r["fs"])
        save_model_plots(model_name, r_nf, r_fs, y_tv, y_te, plot_dir)

    # 1 PNG for the ensemble

    print(f"\n  Total PNGs: {len(MODEL_REGISTRY) * 3 + 1}")


if __name__ == "__main__":
    main()
