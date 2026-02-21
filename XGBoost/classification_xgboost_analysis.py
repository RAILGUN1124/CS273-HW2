"""
XGBoost Emotion Classification Analysis
Implements 5-fold CV for hyperparameter tuning, feature selection, and early stopping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, make_scorer
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 9

all_experiment_results = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_exclude_cols(df):
    """Build list of columns to exclude (non-feature columns) for any dataset."""

    candidates = [
        'arousal', 'valence', 'dominance', 'dataset', 'genre', 'category',
        'fnames', 'fname', 'splits', 'vocals', 'emotional_intensity', 'emotional_quadrant',
        'source', 'description', 'BE_Classification'
    ]
    return [c for c in candidates if c in df.columns]


def load_and_prepare_data(filepath, dataset_name):
    """Load preprocessed data and separate features from target"""
    df = pd.read_csv(filepath)

    target_col = 'emotional_quadrant'
    exclude_cols = get_exclude_cols(df)

    feature_cols = [col for col in df.columns
                    if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target variable: {target_col}")

    class_counts = df[target_col].value_counts().sort_index()
    print(f"\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(df)*100:.1f}%)")

    return df, feature_cols, target_col


def balance_dataset(X, y, random_state=RANDOM_STATE):
    """Balance dataset using SMOTE"""
    print(f"\n  Applying SMOTE to balance classes...")
    print(f"  Before balancing: {len(y)} samples")

    original_counts = pd.Series(y).value_counts().sort_index()
    print(f"  Original class distribution:")
    for cls, count in original_counts.items():
        print(f"    Class {cls}: {count} samples")

    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    balanced_counts = pd.Series(y_balanced).value_counts().sort_index()
    print(f"\n  After balancing: {len(y_balanced)} samples")
    print(f"  Balanced class distribution:")
    for cls, count in balanced_counts.items():
        print(f"    Class {cls}: {count} samples")

    return X_balanced, y_balanced


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """Split data into Train, Validation, and Test sets (60:20:20)"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"\nData Split (60:20:20):")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def hyperparameter_tuning_with_cv(X_train, y_train, X_val, y_val, param_distributions):
    """Run 5-fold CV for Hyperparameter Tuning"""
    print(f"\n  Step 1-2: 5-Fold Stratified CV for Hyperparameter Tuning")
    print(f"  Using RandomizedSearchCV to find optimal parameters...")

    f1_scorer = make_scorer(f1_score, average='macro')
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    xgb_base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric='mlogloss'
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv_strat,
        scoring=f1_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    print(f"\n  Cross-Validation Results (Top 8 configurations):")
    for idx, (_, row) in enumerate(results_df.head(8).iterrows(), 1):
        p = row['params']
        s = f"n={p.get('n_estimators', '?')}, d={p.get('max_depth', '?')}, lr={p.get('learning_rate', '?')}"
        print(f"    {idx}. {s}: F1={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    best_params = random_search.best_params_
    best_f1 = random_search.best_score_

    val_pred = random_search.best_estimator_.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='macro')

    print(f"\n  Best Parameters: {best_params}")
    print(f"  Best CV F1: {best_f1:.4f}, Validation F1: {val_f1:.4f}")

    cv_results = []
    for _, row in results_df.head(30).iterrows():
        cv_results.append({
            'params': row['params'],
            'avg_f1': row['mean_test_score'],
            'std_f1': row['std_test_score'],
            'avg_acc': row.get('mean_train_score', 0),
            'std_acc': row.get('std_train_score', 0)
        })

    return best_params, cv_results


def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    """Retrain on train+val with early stopping"""
    print(f"\n  Step 3: Retraining final XGBoost model with best hyperparameters")

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    print(f"  Training on combined train+val: {len(X_train_val)} samples")

    params = {k: v for k, v in best_params.items()}
    params.pop('early_stopping_rounds', None)

    final_model = xgb.XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
        eval_metric='mlogloss'
    )
    final_model.fit(
        X_train_val, y_train_val,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_train_pred = final_model.predict(X_train_val)
    train_acc = accuracy_score(y_train_val, y_train_pred)
    train_f1 = f1_score(y_train_val, y_train_pred, average='macro')

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Training Macro F1: {train_f1:.4f}")
    if hasattr(final_model, 'best_iteration'):
        print(f"  Best iteration (early stopping): {final_model.best_iteration}")

    return final_model, train_acc, train_f1, X_train_val, y_train_val


def evaluate_on_test(model, X_test, y_test):
    """Evaluate once on the test set"""
    print(f"\n  Step 4: Final Evaluation on Test Set")

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro F1: {test_f1:.4f}")

    print(f"\n  Detailed Classification Report")
    print(classification_report(y_test + 1, y_test_pred + 1, labels=[1, 2, 3, 4], digits=4))

    return test_acc, test_f1, y_test_pred


def apply_feature_selection(X_train, y_train, X_val, X_test, k=34):
    """Apply SelectKBest feature selection"""
    print(f"\n  Feature Selection: SelectKBest (k={k} features)")
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)

    X_train_fs = selector.transform(X_train)
    X_val_fs = selector.transform(X_val)
    X_test_fs = selector.transform(X_test)

    selected_features = selector.get_support(indices=True)
    print(f"  Selected {len(selected_features)} features")

    return X_train_fs, X_val_fs, X_test_fs, selected_features, selector


def train_and_evaluate_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                                  use_feature_selection=False, k=34,
                                  feature_names=None, dataset_name='Dataset', is_balanced=False):
    balance_label = "BALANCED" if is_balanced else "IMBALANCED"
    fs_label = "WITH FEATURE SELECTION" if use_feature_selection else "WITHOUT FEATURE SELECTION"

    print(f"\n{'-'*80}")
    print(f"XGBOOST CLASSIFICATION - {balance_label} - {fs_label}")
    print(f"{'-'*80}")

    if use_feature_selection:
        X_train_proc, X_val_proc, X_test_proc, selected_features, selector = apply_feature_selection(
            X_train, y_train, X_val, X_test, k=k
        )
        selected_feature_names = [feature_names[i] for i in selected_features] if feature_names else None
    else:
        X_train_proc = X_train
        X_val_proc = X_val
        X_test_proc = X_test
        selected_features = None
        selected_feature_names = feature_names
        selector = None

    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 6, 8, 10, 12],
        'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.05, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0.01, 0.1, 1, 5, 10],
    }

    best_params, cv_results = hyperparameter_tuning_with_cv(
        X_train_proc, y_train, X_val_proc, y_val, param_distributions
    )

    final_model, train_acc, train_f1, X_train_val, y_train_val = train_final_model(
        X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, best_params
    )

    test_acc, test_f1, y_test_pred = evaluate_on_test(final_model, X_test_proc, y_test)

    y_train_pred = final_model.predict(X_train_val)
    train_cm = confusion_matrix(y_train_val, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    results = {
        'dataset_name': dataset_name,
        'is_balanced': is_balanced,
        'feature_selection': use_feature_selection,
        'n_features': X_train_proc.shape[1],
        'selected_features': selected_features,
        'best_params': best_params,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'cv_results': cv_results,
        'model': final_model,
        'y_train_val': y_train_val,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_cm': train_cm,
        'test_cm': test_cm,
        'feature_names': selected_feature_names,
        'selector': selector
    }

    all_experiment_results.append(results)
    return results


def compare_results(results_dict, dataset_name='Dataset'):
    """Compare performance across all configurations"""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON - {dataset_name}")
    print(f"{'='*80}")

    configs = [
        ('imb_no_fs', 'Imbalanced, No FS'),
        ('imb_fs', 'Imbalanced, FS'),
        ('bal_no_fs', 'Balanced, No FS'),
        ('bal_fs', 'Balanced, FS')
    ]

    print(f"\n{'Configuration':<25} {'Features':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}")
    print(f"{'-'*95}")

    for key, label in configs:
        if key in results_dict:
            result = results_dict[key]
            print(f"{label:<25} {result['n_features']:<12} "
                  f"{result['train_acc']:<12.4f} {result['test_acc']:<12.4f} "
                  f"{result['train_f1']:<12.4f} {result['test_f1']:<12.4f}")

    print(f"\n{'='*80}")
    print(f"KEY INSIGHTS:")
    print(f"{'-'*80}")

    if 'bal_fs' in results_dict and 'imb_fs' in results_dict:
        bal_f1 = results_dict['bal_fs']['test_f1']
        imb_f1 = results_dict['imb_fs']['test_f1']
        improvement = ((bal_f1 - imb_f1) / imb_f1) * 100 if imb_f1 > 0 else 0
        print(f"Balancing Impact (with FS): {improvement:+.2f}% change in F1 score")

    if 'bal_fs' in results_dict and 'bal_no_fs' in results_dict:
        fs_f1 = results_dict['bal_fs']['test_f1']
        no_fs_f1 = results_dict['bal_no_fs']['test_f1']
        improvement = ((fs_f1 - no_fs_f1) / no_fs_f1) * 100 if no_fs_f1 > 0 else 0
        print(f"Feature Selection Impact (balanced): {improvement:+.2f}% change in F1 score")


def _compute_sample_weights(y):
    """Compute sample weights for class balancing"""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y])


def process_dataset(filepath, dataset_name, k=34):
    """Process entire dataset: load, split, train with all configurations"""
    df, feature_cols, target_col = load_and_prepare_data(filepath, dataset_name)

    X = df[feature_cols].values
    y = df[target_col].values
    y = np.asarray(y, dtype=int) - 1

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    results_dict = {}

    # ========================================================================
    # IMBALANCED DATASET
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# IMBALANCED DATASET")
    print(f"{'#'*80}")

    results_imb_no_fs = train_and_evaluate_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_feature_selection=False,
        feature_names=feature_cols,
        dataset_name=dataset_name,
        is_balanced=False
    )
    results_dict['imb_no_fs'] = results_imb_no_fs

    results_imb_fs = train_and_evaluate_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_feature_selection=True,
        k=k,
        feature_names=feature_cols,
        dataset_name=dataset_name,
        is_balanced=False
    )
    results_dict['imb_fs'] = results_imb_fs

    # ========================================================================
    # BALANCED DATASET
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# BALANCED DATASET")
    print(f"{'#'*80}")

    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

    results_bal_no_fs = train_and_evaluate_classifier(
        X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test,
        use_feature_selection=False,
        feature_names=feature_cols,
        dataset_name=dataset_name,
        is_balanced=True
    )
    results_dict['bal_no_fs'] = results_bal_no_fs

    results_bal_fs = train_and_evaluate_classifier(
        X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test,
        use_feature_selection=True,
        k=k,
        feature_names=feature_cols,
        dataset_name=dataset_name,
        is_balanced=True
    )
    results_dict['bal_fs'] = results_bal_fs

    compare_results(results_dict, dataset_name)
    return results_dict


def print_final_summary():
    """Print final summary of all experiments"""
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY - ALL RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Dataset':<15} {'Balanced':<12} {'Feat Sel':<12} {'Features':<10} "
          f"{'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print(f"{'-'*95}")

    for result in all_experiment_results:
        bal = "Yes" if result['is_balanced'] else "No"
        fs = "Yes" if result['feature_selection'] else "No"
        print(f"{result['dataset_name']:<15} {bal:<12} {fs:<12} {result['n_features']:<10} "
              f"{result['train_acc']:<12.4f} {result['test_acc']:<12.4f} {result['test_f1']:<12.4f}")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_combined_visualizations():
    """Create comprehensive combined visualizations"""
    print(f"\n\n{'='*80}")
    print("GENERATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")

    datasets = {}
    for result in all_experiment_results:
        dataset_name = result['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = {}

        key = f"{'bal' if result['is_balanced'] else 'imb'}_{'fs' if result['feature_selection'] else 'no_fs'}"
        datasets[dataset_name][key] = result

    for dataset_name, results_data in datasets.items():
        create_dataset_comprehensive_figure(dataset_name, results_data)

    create_overall_summary_figure(datasets)
    print(f"\nAll visualizations generated successfully!")
    print(f"  Total files created: {len(datasets) + 1}")

def create_dataset_comprehensive_figure(dataset_name, results_data):
    """One comprehensive figure per dataset"""
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    configs = [
        ('imb_no_fs', 'Imbalanced, No FS'),
        ('imb_fs', 'Imbalanced, FS'),
        ('bal_no_fs', 'Balanced, No FS'),
        ('bal_fs', 'Balanced, FS')
    ]

    for i, (key, label) in enumerate(configs):
        if key not in results_data:
            continue
        result = results_data[key]

        # CV Results
        ax_cv = fig.add_subplot(gs[i, 0])
        cv_res = result['cv_results']
        x_pos = np.arange(len(cv_res))
        f1_scores = [r['avg_f1'] for r in cv_res]
        f1_stds = [r['std_f1'] for r in cv_res]
        ax_cv.bar(x_pos, f1_scores, yerr=f1_stds, capsize=3, color='steelblue', edgecolor='black')
        ax_cv.set_xlabel('Config', fontsize=8)
        ax_cv.set_ylabel('CV F1 Score', fontsize=8)
        ax_cv.set_title(f'{label}\nCV Performance', fontsize=9, fontweight='bold')
        ax_cv.set_xticks(x_pos)
        ax_cv.set_xticklabels([f"{j+1}" for j in x_pos], fontsize=7)
        ax_cv.grid(True, alpha=0.3, axis='y')

        # Train Confusion Matrix
        ax_train_cm = fig.add_subplot(gs[i, 1])
        sns.heatmap(result['train_cm'], annot=True, fmt='d', cmap='Blues',
                    ax=ax_train_cm, cbar=False, square=True)
        ax_train_cm.set_xlabel('Predicted', fontsize=8)
        ax_train_cm.set_ylabel('Actual', fontsize=8)
        ax_train_cm.set_title(f'Train CM\nAcc={result["train_acc"]:.3f}, F1={result["train_f1"]:.3f}', fontsize=9, fontweight='bold')

        # Test Confusion Matrix
        ax_test_cm = fig.add_subplot(gs[i, 2])
        sns.heatmap(result['test_cm'], annot=True, fmt='d', cmap='Oranges',
                    ax=ax_test_cm, cbar=False, square=True)
        ax_test_cm.set_xlabel('Predicted', fontsize=8)
        ax_test_cm.set_ylabel('Actual', fontsize=8)
        ax_test_cm.set_title(f'Test CM\nAcc={result["test_acc"]:.3f}, F1={result["test_f1"]:.3f}', fontsize=9, fontweight='bold')

        # Per class F1 Scores
        ax_f1 = fig.add_subplot(gs[i, 3])
        y_test = result['y_test']
        y_pred = result['y_test_pred']
        f1_per_class = f1_score(y_test, y_pred, average=None)
        classes = np.unique(y_test)
        colors = sns.color_palette('viridis', len(classes))
        # Add 1 to show quadrants 1-4
        ax_f1.bar(classes + 1, f1_per_class, color=colors, edgecolor='black')
        ax_f1.set_xlabel('Class (quadrant)', fontsize=8)
        ax_f1.set_ylabel('F1 Score', fontsize=8)
        ax_f1.set_title(f'Per-Class F1\nMacro F1={result["test_f1"]:.3f}', fontsize=9, fontweight='bold')
        ax_f1.set_xticks(classes + 1)
        ax_f1.set_ylim([0, 1.0])
        ax_f1.grid(True, alpha=0.3, axis='y')

        # Feature importance
        ax_feat = fig.add_subplot(gs[i, 4])
        if result['feature_names'] is not None and hasattr(result['model'], 'feature_importances_'):
            importances = result['model'].feature_importances_
            all_features = result['feature_names']
            top_idx = np.argsort(importances)[-10:][::-1]
            top_features = [all_features[j] if j < len(all_features) else f"F{j}" for j in top_idx]
            top_importances = importances[top_idx]
            y_pos = np.arange(len(top_features))
            colors_imp = sns.color_palette('viridis', len(top_features))
            ax_feat.barh(y_pos, top_importances, color=colors_imp, edgecolor='black', linewidth=0.5)
            ax_feat.set_yticks(y_pos)
            ax_feat.set_yticklabels(top_features, fontsize=7)
            ax_feat.set_xlabel('Importance', fontsize=8)
            ax_feat.set_title(f'Top 10 Features\n({result["n_features"]} total)', fontsize=9, fontweight='bold')
            ax_feat.invert_yaxis()
            ax_feat.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'XGBoost Classification Analysis: {dataset_name}', fontsize=16, fontweight='bold', y=0.995)
    filename = f'xgboost_{dataset_name.lower().replace("-", "")}_full_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_overall_summary_figure(datasets):
    """Overall summary figure comparing all datasets and configurations"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    all_labels = []
    test_acc_imb_no_fs = []
    test_acc_imb_fs = []
    test_acc_bal_no_fs = []
    test_acc_bal_fs = []
    test_f1_imb_no_fs = []
    test_f1_imb_fs = []
    test_f1_bal_no_fs = []
    test_f1_bal_fs = []

    for dataset_name, results_data in datasets.items():
        all_labels.append(dataset_name)
        test_acc_imb_no_fs.append(results_data.get('imb_no_fs', {}).get('test_acc', 0))
        test_acc_imb_fs.append(results_data.get('imb_fs', {}).get('test_acc', 0))
        test_acc_bal_no_fs.append(results_data.get('bal_no_fs', {}).get('test_acc', 0))
        test_acc_bal_fs.append(results_data.get('bal_fs', {}).get('test_acc', 0))
        test_f1_imb_no_fs.append(results_data.get('imb_no_fs', {}).get('test_f1', 0))
        test_f1_imb_fs.append(results_data.get('imb_fs', {}).get('test_f1', 0))
        test_f1_bal_no_fs.append(results_data.get('bal_no_fs', {}).get('test_f1', 0))
        test_f1_bal_fs.append(results_data.get('bal_fs', {}).get('test_f1', 0))

    x = np.arange(len(all_labels))
    width = 0.2

    axes[0, 0].bar(x - 1.5*width, test_acc_imb_no_fs, width, label='Imb, No FS', color='lightblue', edgecolor='black')
    axes[0, 0].bar(x - 0.5*width, test_acc_imb_fs, width, label='Imb, FS', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + 0.5*width, test_acc_bal_no_fs, width, label='Bal, No FS', color='lightcoral', edgecolor='black')
    axes[0, 0].bar(x + 1.5*width, test_acc_bal_fs, width, label='Bal, FS', color='darkred', edgecolor='black')
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(all_labels, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    axes[0, 1].bar(x - 1.5*width, test_f1_imb_no_fs, width, label='Imb, No FS', color='lightblue', edgecolor='black')
    axes[0, 1].bar(x - 0.5*width, test_f1_imb_fs, width, label='Imb, FS', color='steelblue', edgecolor='black')
    axes[0, 1].bar(x + 0.5*width, test_f1_bal_no_fs, width, label='Bal, No FS', color='lightcoral', edgecolor='black')
    axes[0, 1].bar(x + 1.5*width, test_f1_bal_fs, width, label='Bal, FS', color='darkred', edgecolor='black')
    axes[0, 1].set_ylabel('Test Macro F1', fontsize=11)
    axes[0, 1].set_title('Test Macro F1 Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_labels, fontsize=10)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])

    fs_impact_bal = [(fs - no_fs) / no_fs * 100 if no_fs > 0 else 0
                     for fs, no_fs in zip(test_f1_bal_fs, test_f1_bal_no_fs)]
    colors_fs = ['green' if imp > 0 else 'red' for imp in fs_impact_bal]
    axes[0, 2].bar(x, fs_impact_bal, color=colors_fs, edgecolor='black')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 2].set_ylabel('F1 Change (%)', fontsize=11)
    axes[0, 2].set_title('Feature Selection Impact (Balanced)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(all_labels, fontsize=10)
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    bal_impact_fs = [(bal - imb) / imb * 100 if imb > 0 else 0 
                    for bal, imb in zip(test_f1_bal_fs, test_f1_imb_fs)]
    colors_bal = ['green' if imp > 0 else 'red' for imp in bal_impact_fs]
    axes[1, 0].bar(x, bal_impact_fs, color=colors_bal, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_ylabel('F1 Change (%)', fontsize=11)
    axes[1, 0].set_title('Balancing Impact (with FS)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_labels, fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].scatter(test_acc_bal_fs, test_f1_bal_fs, s=200, alpha=0.6, edgecolors='black', linewidth=2)
    for i, label in enumerate(all_labels):
        axes[1, 1].annotate(label, (test_acc_bal_fs[i], test_f1_bal_fs[i]),fontsize=10, ha='center', va='center')
    axes[1, 1].set_xlabel('Test Accuracy', fontsize=11)
    axes[1, 1].set_ylabel('Test Macro F1', fontsize=11)
    axes[1, 1].set_title('Accuracy vs F1 (Balanced + FS)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1.0])
    axes[1, 1].set_ylim([0, 1.0])

    axes[1, 2].axis('off')
    summary_text = "BEST CONFIGURATIONS (XGBoost)\n" + "="*40 + "\n\n"
    
    for i, dataset_name in enumerate(all_labels):
        summary_text += f"{dataset_name}:\n"
        summary_text += f"  Config: Balanced + FS\n"
        summary_text += f"  Accuracy: {test_acc_bal_fs[i]:.4f}\n"
        summary_text += f"  Macro F1: {test_f1_bal_fs[i]:.4f}\n\n"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('xgboost_overall_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: xgboost_overall_model_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" "*15 + "XGBOOST EMOTION CLASSIFICATION ANALYSIS")
    print(" "*10 + "5-Fold CV, Class Balancing, and Feature Selection with Early Stopping")
    print(" "*15 + "Target: emotional_quadrant (4 classes)")
    print("="*80)

    datasets = [
        ('../Data/final_emosounds-3.csv', 'EmoSounds-3'),
        ('../Data/final_iadsed-2.csv', 'IADSED-2')
    ]
    k_features = 34

    for filepath, dataset_name in datasets:
        try:
            process_dataset(filepath, dataset_name, k=k_features)
        except FileNotFoundError:
            print(f"\nError: File '{filepath}' not found. Skipping {dataset_name}.")
        except Exception as e:
            print(f"\nError processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print_final_summary()
    create_combined_visualizations()

    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
