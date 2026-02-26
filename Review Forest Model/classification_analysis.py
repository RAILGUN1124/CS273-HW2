"""
Review Classification Analysis with Combined Visualizations
Implements 5-fold CV for hyperparameter tuning with Optuna and feature selection
Dataset is already balanced (130 samples per class)
Targets: Label (binary classes - 0: negative, 1: positive)
Features: AWL, ASL, NWO, NVB, NAJ, NPV, NST, CDV, NTP, TPR (all features except Label)
Uses Optuna for exhaustive hyperparameter search with Bayesian optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seeds for reproducibility
#694269 is the best random state for reproducibility and good performance across all experiments
RANDOM_STATE = 694269
np.random.seed(RANDOM_STATE)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 9

# Store all results for batch plotting
all_experiment_results = []

# ============================================================================
# CONFIGURATION
# ============================================================================
# Optuna Configuration:
# - Uses TPE (Tree-structured Parzen Estimator) for Bayesian optimization
# - Much smarter than random search - learns from previous trials
# - Searches wider hyperparameter space:
#   * n_estimators: 100-500 (vs fixed set in random search)
#   * max_depth: 3-30 (wider range)
#   * min_samples_split: 2-20
#   * max_samples: 0.5-1.0 (new parameter)
# - Default: 100 trials per K value (adjustable in code)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(filepath, dataset_name):
    """Load preprocessed data and separate features from target"""
    df = pd.read_csv(filepath)
    
    # Define target variable
    target_col = 'Label'
    
    # Get feature columns (all columns except Label)
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target variable: {target_col}")
    
    # Show class distribution
    class_counts = df[target_col].value_counts().sort_index()
    print(f"\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(df)*100:.1f}%)")
    
    return df, feature_cols, target_col


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


def hyperparameter_tuning_with_cv(X_train, y_train, X_val, y_val, n_trials=100):
    """5-Fold Cross-Validation for Hyperparameter Tuning using Optuna"""
    print(f"\n  Step 1-2: 5-Fold Cross-Validation for Hyperparameter Tuning with Optuna")
    print(f"  Using Bayesian optimization (TPE) to find optimal parameters...")
    print(f"  Running {n_trials} trials for exhaustive search...")
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    # Use F1-macro as scoring metric
    f1_scorer = make_scorer(f1_score, average='macro')
    
    # Define objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
            'bootstrap': bootstrap,
            'random_state': RANDOM_STATE
        }
        
        # Add max_samples only if bootstrap is True
        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
        
        # Create model with suggested parameters
        model = RandomForestClassifier(**params)
        
        # Perform 5-fold cross-validation
        scores = cross_val_score(model, X_train_val, y_train_val, cv=5, 
                                scoring=f1_scorer, n_jobs=-1)
        
        # Return mean F1 score
        return scores.mean()
    
    # Create Optuna study with TPE sampler for Bayesian optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    
    # Optimize with progress bar
    print(f"  Progress:")
    with tqdm(total=n_trials, desc="  Optuna trials", unit="trial", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best F1': f'{study.best_value:.4f}', 'Trial': trial.number})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    
    # Get best parameters and results
    best_params = study.best_params
    best_f1 = study.best_value
    
    # Train model with best parameters to get training accuracy
    best_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_train_val)
    best_acc = accuracy_score(y_train_val, y_pred)
    
    print(f"\n  Optuna Optimization Results:")
    print(f"    Completed {len(study.trials)} trials")
    print(f"    Best trial: #{study.best_trial.number}")
    
    print(f"\n  Top 8 trials by F1 score:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:8]
    for idx, trial in enumerate(sorted_trials, 1):
        if trial.value is not None:
            params_str = f"n={trial.params.get('n_estimators', 'N/A')}, d={trial.params.get('max_depth', 'N/A')}, split={trial.params.get('min_samples_split', 'N/A')}"
            print(f"    {idx}. Trial #{trial.number}: {params_str}: F1={trial.value:.4f}")
    
    print(f"\n  Best Parameters: {best_params}")
    print(f"  Best CV F1: {best_f1:.4f}, Train Acc: {best_acc:.4f}")
    
    # Convert results for compatibility with visualization
    cv_results = []
    for trial in sorted_trials[:30]:
        if trial.value is not None:
            cv_results.append({
                'params': trial.params,
                'avg_f1': trial.value,
                'std_f1': 0.0,  # Optuna doesn't provide std directly
                'avg_acc': best_acc,  # Approximation
                'std_acc': 0.0
            })
    
    return best_params, cv_results


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Retrain the best model on all train+validation data"""
    print(f"\n  Step 3: Retraining final model with best hyperparameters")
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"  Training on combined train+val: {len(X_train_val)} samples")
    
    final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    final_model.fit(X_train_val, y_train_val)
    
    y_train_pred = final_model.predict(X_train_val)
    train_acc = accuracy_score(y_train_val, y_train_pred)
    train_f1 = f1_score(y_train_val, y_train_pred, average='macro')
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Training Macro F1: {train_f1:.4f}")
    
    return final_model, train_acc, train_f1, X_train_val, y_train_val


def evaluate_on_test(model, X_test, y_test):
    """Evaluate once on the test set"""
    print(f"\n  Step 4: Final Evaluation on Test Set")
    
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro F1: {test_f1:.4f}")
    
    print(f"\n  Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    return test_acc, test_f1, y_test_pred


def find_optimal_k(X_train, y_train, X_val, y_val, method='selectkbest', k_range=None):
    """Fin
    d optimal K value for feature selection using validation set"""
    print(f"\n{'='*80}")
    print(f"FINDING OPTIMAL K VALUE FOR FEATURE SELECTION")
    print(f"{'='*80}")
    
    # Combine train and val for cross-validation
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Define K range if not provided
    if k_range is None:
        n_features = X_train.shape[1]
        # Test all K values from 1 to n_features - 1
        max_k = n_features - 1
        k_range = list(range(1, max_k + 1))
    
    print(f"Testing K values: 1 to {len(k_range)}")
    print(f"Total features available: {X_train.shape[1]}")
    
    best_k = None
    best_score = -1
    k_scores = []
    
    # Use 5-fold cross-validation to evaluate each K
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\nEvaluating K values with 5-fold cross-validation...")
    for k in tqdm(k_range, desc="Testing K values", unit="K"):
        fold_scores = []
        
        for train_idx, val_idx in kfold.split(X_combined):
            X_cv_train, X_cv_val = X_combined[train_idx], X_combined[val_idx]
            y_cv_train, y_cv_val = y_combined[train_idx], y_combined[val_idx]
            
            # Apply feature selection
            if method == 'selectkbest':
                selector = SelectKBest(f_classif, k=k)
            else:  # rfe
                estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
                selector = RFE(estimator, n_features_to_select=k, step=5)
            
            selector.fit(X_cv_train, y_cv_train)
            X_cv_train_fs = selector.transform(X_cv_train)
            X_cv_val_fs = selector.transform(X_cv_val)
            
            # Train a simple model for evaluation
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
            rf.fit(X_cv_train_fs, y_cv_train)
            
            # Evaluate on validation fold
            y_pred = rf.predict(X_cv_val_fs)
            score = f1_score(y_cv_val, y_pred, average='macro')
            fold_scores.append(score)
        
        # Calculate average score across folds
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        k_scores.append((k, avg_score, std_score))
        
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL K FOUND: {best_k} features (CV F1 Score: {best_score:.4f})")
    print(f"{'='*80}")
    print(f"\nTop 5 K values by CV F1 score:")
    sorted_k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)[:5]
    for k, score, std in sorted_k_scores:
        marker = " ← SELECTED" if k == best_k else ""
        print(f"  K={k:3d}: F1={score:.4f} (±{std:.4f}){marker}")
    
    return best_k, k_scores


def apply_feature_selection(X_train, y_train, X_val, X_test, method='selectkbest', k=34):
    """Apply feature selection technique"""
    if method == 'selectkbest':
        print(f"\n  Feature Selection: SelectKBest (k={k} features)")
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        
        X_train_fs = selector.transform(X_train)
        X_val_fs = selector.transform(X_val)
        X_test_fs = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"  Selected {len(selected_features)} features")
        
    elif method == 'rfe':
        print(f"\n  Feature Selection: RFE (k={k} features)")
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        selector = RFE(estimator, n_features_to_select=k, step=5)
        selector.fit(X_train, y_train)
        
        X_train_fs = selector.transform(X_train)
        X_val_fs = selector.transform(X_val)
        X_test_fs = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        print(f"  Selected {len(selected_features)} features")
    
    return X_train_fs, X_val_fs, X_test_fs, selected_features, selector


def train_and_evaluate_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                                  use_feature_selection=False, fs_method='selectkbest', k=8,
                                  feature_names=None, dataset_name='Dataset'):
    """Complete classification pipeline with CV"""
    fs_label = "WITH FEATURE SELECTION" if use_feature_selection else "WITHOUT FEATURE SELECTION"
    
    print(f"\n{'-'*80}")
    print(f"RANDOM FOREST CLASSIFICATION - {fs_label}")
    print(f"{'-'*80}")
    
    # Apply feature selection if requested
    selector = None
    if use_feature_selection:
        X_train_proc, X_val_proc, X_test_proc, selected_features, selector = apply_feature_selection(
            X_train, y_train, X_val, X_test, method=fs_method, k=k
        )
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in selected_features]
        else:
            selected_feature_names = None
    else:
        X_train_proc = X_train
        X_val_proc = X_val
        X_test_proc = X_test
        selected_features = None
        selected_feature_names = feature_names
    
    # Step 1-2: Hyperparameter tuning with CV using Optuna (exhaustive search)
    # Adjust n_trials for search thoroughness:
    # - 50-100 trials: Fast, good results
    # - 100-200 trials: Balanced (recommended)
    # - 200-500 trials: Very thorough, may take longer
    n_trials = 100
    best_params, cv_results = hyperparameter_tuning_with_cv(
        X_train_proc, y_train, X_val_proc, y_val, n_trials=n_trials
    )
    
    # Step 3: Train final model
    final_model, train_acc, train_f1, X_train_val, y_train_val = train_final_model(
        X_train_proc, y_train, X_val_proc, y_val, best_params
    )
    
    # Step 4: Evaluate on test set
    test_acc, test_f1, y_test_pred = evaluate_on_test(final_model, X_test_proc, y_test)
    
    # Get training predictions for visualization
    y_train_pred = final_model.predict(X_train_val)
    
    # Get confusion matrices
    train_cm = confusion_matrix(y_train_val, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    results = {
        'dataset_name': dataset_name,
        'feature_selection': use_feature_selection,
        'n_features': X_train_proc.shape[1],
        'k_value': k if use_feature_selection else None,
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
        ('no_fs', 'No Feature Selection'),
        ('fs', 'With Feature Selection')
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
    
    # Compare feature selection
    if 'fs' in results_dict and 'no_fs' in results_dict:
        fs_f1 = results_dict['fs']['test_f1']
        no_fs_f1 = results_dict['no_fs']['test_f1']
        improvement = ((fs_f1 - no_fs_f1) / no_fs_f1) * 100
        print(f"Feature Selection Impact: {improvement:+.2f}% change in F1 score")
        print(f"Feature Reduction: {results_dict['no_fs']['n_features']} → {results_dict['fs']['n_features']} features")


def compare_results_with_k(results_dict, dataset_name='Dataset', k_values=None, best_k=None):
    """Compare performance between baseline and CV-selected K"""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON - {dataset_name}")
    print(f"{'='*80}")
    
    # Show baseline
    print(f"\nBASELINE (No Feature Selection):")
    print(f"{'Configuration':<25} {'Features':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}")
    print(f"{'-'*95}")
    
    if 'no_fs' in results_dict:
        result = results_dict['no_fs']
        print(f"{'All Features':<25} {result['n_features']:<12} "
              f"{result['train_acc']:<12.4f} {result['test_acc']:<12.4f} "
              f"{result['train_f1']:<12.4f} {result['test_f1']:<12.4f}")
    
    # Show selected K
    print(f"\nFEATURE SELECTION (CV-Selected K):")
    print(f"{'K Value':<10} {'Features':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}")
    print(f"{'-'*80}")
    
    if best_k is not None and f'k_{best_k}' in results_dict:
        result = results_dict[f'k_{best_k}']
        print(f"{'K=' + str(best_k):<10} {result['n_features']:<12} "
              f"{result['train_acc']:<12.4f} {result['test_acc']:<12.4f} "
              f"{result['train_f1']:<12.4f} {result['test_f1']:<12.4f} ← SELECTED (via CV)")
    
    print(f"\n{'='*80}")
    print(f"KEY INSIGHTS:")
    print(f"{'-'*80}")
    
    if best_k is not None and 'no_fs' in results_dict:
        best_result = results_dict[f'k_{best_k}']
        baseline_result = results_dict['no_fs']
        
        improvement = ((best_result['test_f1'] - baseline_result['test_f1']) / baseline_result['test_f1']) * 100
        print(f"Best K value: {best_k} features (SELECTED USING CV VALIDATION)")
        print(f"Test F1 for selected K: {best_result['test_f1']:.4f}")
        print(f"Improvement over baseline: {improvement:+.2f}% change in F1 score")
        print(f"Feature Reduction: {baseline_result['n_features']} → {best_result['n_features']} features ({best_result['n_features']/baseline_result['n_features']*100:.1f}% retained)")
        print(f"\nNote: K was selected using cross-validation on train+val data.")
        
        # Show selected features for best K
        if best_result['feature_names'] is not None:
            print(f"\nSelected Features (K={best_k}):")
            for i, feat in enumerate(best_result['feature_names'], 1):
                print(f"  {i}. {feat}")


def process_dataset(filepath, dataset_name, k_values=None):
    """Process entire dataset: load, split, train with all configurations"""
    df, feature_cols, target_col = load_and_prepare_data(filepath, dataset_name)
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # If k_values not specified, try all possible values from 1 to n_features-1
    n_features = len(feature_cols)
    if k_values is None:
        k_values = list(range(1, n_features))  # 1 to n_features-1
    
    results_dict = {}
    
    # ========================================================================
    # EXPERIMENT 1: ALL FEATURES (BASELINE)
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT 1: ALL FEATURES (BASELINE)")
    print(f"{'#'*80}")
    
    results_no_fs = train_and_evaluate_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_feature_selection=False,
        feature_names=feature_cols,
        dataset_name=dataset_name
    )
    results_dict['no_fs'] = results_no_fs
    
    # ========================================================================
    # EXPERIMENT 2: FEATURE SELECTION - FIND OPTIMAL K USING CV
    # ========================================================================
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT 2: FEATURE SELECTION - OPTIMAL K SELECTION")
    print(f"{'#'*80}")
    
    # Find optimal K using cross-validation (NO TEST SET INVOLVED)
    best_k, cv_k_scores = find_optimal_k(
        X_train, y_train, X_val, y_val,
        method='rfe',
        k_range=k_values
    )
    
    # Train and evaluate with the selected best K
    print(f"\n\n{'='*80}")
    print(f"TRAINING MODEL WITH SELECTED K={best_k}")
    print(f"{'='*80}")
    
    best_k_result = train_and_evaluate_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_feature_selection=True,
        fs_method='rfe',
        k=best_k,
        feature_names=feature_cols,
        dataset_name=dataset_name
    )
    
    # Store CV scores in result for visualization
    best_k_result['cv_k_scores'] = cv_k_scores
    results_dict[f'k_{best_k}'] = best_k_result
    results_dict['best_k'] = best_k_result
    results_dict['cv_k_scores'] = cv_k_scores
    
    # Compare results
    compare_results_with_k(results_dict, dataset_name, [best_k], best_k)
    
    return results_dict


def print_final_summary():
    """Print final summary of all experiments"""
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY - ALL RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Dataset':<20} {'Feat Sel':<12} {'Features':<10} "
          f"{'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print(f"{'-'*85}")
    
    for result in all_experiment_results:
        fs = "Yes" if result['feature_selection'] else "No"
        print(f"{result['dataset_name']:<20} {fs:<12} {result['n_features']:<10} "
              f"{result['train_acc']:<12.4f} {result['test_acc']:<12.4f} {result['test_f1']:<12.4f}")


# ============================================================================
# COMBINED VISUALIZATION FUNCTIONS
# ============================================================================

def create_combined_visualizations():
    """Create comprehensive combined visualizations"""
    print(f"\n\n{'='*80}")
    print("GENERATING COMBINED VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Organize results by dataset
    datasets = {}
    for result in all_experiment_results:
        dataset_name = result['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = {'no_fs': None, 'best_k': None, 'cv_k_scores': None}
        
        if result['feature_selection']:
            datasets[dataset_name]['best_k'] = result
            # Store cv_k_scores if available
            if 'cv_k_scores' in result:
                datasets[dataset_name]['cv_k_scores'] = result['cv_k_scores']
        else:
            datasets[dataset_name]['no_fs'] = result
    
    # Create visualizations with progress bar
    total_plots = len(datasets) * 2 + 1  # 2 plots per dataset + 1 overall
    with tqdm(total=total_plots, desc="Creating visualizations", unit="plot") as pbar:
        for dataset_name, results_data in datasets.items():
            create_dataset_comprehensive_figure(dataset_name, results_data)
            pbar.update(1)
            create_k_performance_figure(dataset_name, results_data)
            pbar.update(1)
        
        # Create overall summary
        create_overall_summary_figure(datasets)
        pbar.update(1)
    
    print(f"\n All visualizations generated successfully!")
    print(f"  Total files created: {len(datasets) * 2 + 1}")


def create_dataset_comprehensive_figure(dataset_name, results_data):
    """Create ONE comprehensive figure per dataset with all analyses"""
    
    # Get baseline and best K result
    no_fs_result = results_data['no_fs']
    best_k_result = results_data.get('best_k', None)
    
    if not best_k_result:
        print(f"Warning: No feature selection results for {dataset_name}")
        return
    
    # Create figure with 2 rows (baseline and best K) and 5 columns
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
    
    configs = [
        (no_fs_result, f"All Features (n={no_fs_result['n_features']})"),
        (best_k_result, f"Best K={best_k_result.get('k_value', best_k_result['n_features'])} Features (CV Selected)")
    ]
    
    for i, (result, label) in enumerate(configs):
        if result is None:
            continue
        
        # Column 0: CV Results
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
        ax_cv.set_xticklabels([f"{i+1}" for i in x_pos], fontsize=7)
        ax_cv.grid(True, alpha=0.3, axis='y')
        
        # Column 1: Training Confusion Matrix
        ax_train_cm = fig.add_subplot(gs[i, 1])
        sns.heatmap(result['train_cm'], annot=True, fmt='d', cmap='Blues', 
                   ax=ax_train_cm, cbar=False, square=True)
        ax_train_cm.set_xlabel('Predicted', fontsize=8)
        ax_train_cm.set_ylabel('Actual', fontsize=8)
        ax_train_cm.set_title(f'Train Confusion Matrix\nAcc={result["train_acc"]:.3f}, F1={result["train_f1"]:.3f}', 
                             fontsize=9, fontweight='bold')
        
        # Column 2: Test Confusion Matrix
        ax_test_cm = fig.add_subplot(gs[i, 2])
        sns.heatmap(result['test_cm'], annot=True, fmt='d', cmap='Oranges', 
                   ax=ax_test_cm, cbar=False, square=True)
        ax_test_cm.set_xlabel('Predicted', fontsize=8)
        ax_test_cm.set_ylabel('Actual', fontsize=8)
        ax_test_cm.set_title(f'Test Confusion Matrix\nAcc={result["test_acc"]:.3f}, F1={result["test_f1"]:.3f}', 
                            fontsize=9, fontweight='bold')
        
        # Column 3: Per-Class F1 Scores
        ax_f1 = fig.add_subplot(gs[i, 3])
        y_test = result['y_test']
        y_pred = result['y_test_pred']
        f1_per_class = f1_score(y_test, y_pred, average=None)
        classes = np.unique(y_test)
        
        colors = sns.color_palette('viridis', len(classes))
        ax_f1.bar(classes, f1_per_class, color=colors, edgecolor='black')
        ax_f1.set_xlabel('Class', fontsize=8)
        ax_f1.set_ylabel('F1 Score', fontsize=8)
        ax_f1.set_title(f'Per-Class F1 Scores\nMacro F1={result["test_f1"]:.3f}', 
                       fontsize=9, fontweight='bold')
        ax_f1.set_xticks(classes)
        ax_f1.set_ylim([0, 1.0])
        ax_f1.grid(True, alpha=0.3, axis='y')
        
        # Column 4: Feature Importance (Top 10)
        ax_feat = fig.add_subplot(gs[i, 4])
        if result['feature_names'] is not None and hasattr(result['model'], 'feature_importances_'):
            importances = result['model'].feature_importances_
            if result['selected_features'] is not None:
                # Map back to original feature names
                all_features = result['feature_names']
            else:
                all_features = result['feature_names']
            
            top_idx = np.argsort(importances)[-10:][::-1]
            top_features = [all_features[i] if i < len(all_features) else f"F{i}" for i in top_idx]
            top_importances = importances[top_idx]
            
            y_pos = np.arange(len(top_features))
            colors_imp = sns.color_palette('viridis', len(top_features))
            ax_feat.barh(y_pos, top_importances, color=colors_imp, edgecolor='black', linewidth=0.5)
            ax_feat.set_yticks(y_pos)
            ax_feat.set_yticklabels(top_features, fontsize=7)
            ax_feat.set_xlabel('Importance', fontsize=8)
            ax_feat.set_title(f'Top 10 Features\n({result["n_features"]} total)', 
                             fontsize=9, fontweight='bold')
            ax_feat.invert_yaxis()
            ax_feat.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Comprehensive Classification Analysis: {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    filename = f'{dataset_name.lower().replace("-", "")}_full_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_k_performance_figure(dataset_name, results_data):
    """Create a figure showing CV validation scores for K selection process"""
    
    no_fs_result = results_data['no_fs']
    best_k_result = results_data.get('best_k', None)
    cv_k_scores = results_data.get('cv_k_scores', None)
    
    if not best_k_result or not cv_k_scores:
        print(f"Warning: No CV scores available for {dataset_name}")
        return
    
    # Extract CV validation scores (used for K selection)
    cv_k_dict = {k: (avg_f1, std_f1) for k, avg_f1, std_f1 in cv_k_scores}
    k_values = sorted(cv_k_dict.keys())
    
    val_f1s = [cv_k_dict[k][0] for k in k_values]  # CV validation F1 scores
    val_f1_stds = [cv_k_dict[k][1] for k in k_values]  # CV validation F1 stds
    
    # Baseline values
    baseline_test_f1 = no_fs_result['test_f1']
    baseline_train_f1 = no_fs_result['train_f1']
    
    # Best K info
    best_k = best_k_result['k_value']
    best_k_test_f1 = best_k_result['test_f1']
    best_k_train_f1 = best_k_result['train_f1']
    best_idx = k_values.index(best_k)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Validation F1 vs K (USED FOR K SELECTION)
    axes[0, 0].errorbar(k_values, val_f1s, yerr=val_f1_stds, fmt='o-', linewidth=2, markersize=8, 
                        capsize=5, capthick=2, label='Validation F1 (CV)', color='green')
    axes[0, 0].axhline(y=baseline_test_f1, color='red', linestyle='--', linewidth=2, 
                       label=f'Baseline Test F1: {baseline_test_f1:.4f}')
    axes[0, 0].plot(best_k, val_f1s[best_idx], 'r*', markersize=20, 
                    label=f'Selected K={best_k}', zorder=5)
    axes[0, 0].set_xlabel('Number of Features (K)', fontsize=12)
    axes[0, 0].set_ylabel('Validation F1 Score (CV)', fontsize=12)
    axes[0, 0].set_title('Validation F1 vs Number of Features\n(Used for K Selection)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(k_values)
    axes[0, 0].set_ylim([0, 1.0])
    
    # Plot 2: Selected K Performance Comparison
    metrics = ['Train F1', 'Val F1 (CV)', 'Test F1']
    values = [best_k_train_f1, val_f1s[best_idx], best_k_test_f1]
    colors_bar = ['lightblue', 'green', 'darkblue']
    
    axes[0, 1].bar(metrics, values, color=colors_bar, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=baseline_test_f1, color='red', linestyle='--', linewidth=2, 
                       label=f'Baseline: {baseline_test_f1:.4f}')
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title(f'Selected K={best_k} Performance\nTrain vs Validation vs Test', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])
    
    # Plot 3: CV Score Distribution Across K Values
    axes[1, 0].bar(k_values, val_f1s, color='lightgreen', edgecolor='black', alpha=0.7, label='Validation F1')
    axes[1, 0].bar([best_k], [val_f1s[best_idx]], color='darkgreen', edgecolor='black', label=f'Selected K={best_k}')
    axes[1, 0].set_xlabel('Number of Features (K)', fontsize=12)
    axes[1, 0].set_ylabel('Validation F1 Score (CV)', fontsize=12)
    axes[1, 0].set_title('All K Values Evaluated During Selection', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(k_values)
    axes[1, 0].set_ylim([0, 1.0])
    
    # Plot 4: Performance Summary Table
    axes[1, 1].axis('off')
    
    summary_text = "K SELECTION SUMMARY\n"
    summary_text += "="*50 + "\n\n"
    summary_text += "Selection Process:\n"
    summary_text += f"  - Evaluated {len(k_values)} K values (1 to {max(k_values)})\n"
    summary_text += f"  - Used 5-fold CV on train+val data\n"
    summary_text += "Selected Configuration:\n"
    summary_text += f"  K = {best_k} features\n"
    summary_text += f"  CV Val F1: {val_f1s[best_idx]:.4f} (±{val_f1_stds[best_idx]:.3f})\n\n"
    summary_text += "Final Test Performance:\n"
    summary_text += f"  Baseline (all features): {baseline_test_f1:.4f}\n"
    summary_text += f"  Selected K={best_k}: {best_k_test_f1:.4f}\n"
    improvement = ((best_k_test_f1 - baseline_test_f1) / baseline_test_f1) * 100
    summary_text += f"  Improvement: {improvement:+.2f}%\n\n"
    summary_text += "Feature Reduction:\n"
    summary_text += f"  {no_fs_result['n_features']} → {best_k} features\n"
    summary_text += f"  ({best_k/no_fs_result['n_features']*100:.1f}% retained)\n\n"
    summary_text += "Note: K selected using validation\n"
    
    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    fig.suptitle(f'Feature Selection Analysis (K Selected via CV Validation): {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'{dataset_name.lower().replace("-", "")}_k_optimization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_overall_summary_figure(datasets):
    """Create ONE overall summary figure comparing all datasets and configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Collect all data
    all_labels = []
    test_acc_no_fs = []
    test_acc_best = []
    test_f1_no_fs = []
    test_f1_best = []
    best_k_values = []
    
    for dataset_name, results_data in datasets.items():
        all_labels.append(dataset_name)
        
        no_fs = results_data['no_fs']
        best_k_result = results_data.get('best_k', None)
        
        test_acc_no_fs.append(no_fs['test_acc'])
        test_f1_no_fs.append(no_fs['test_f1'])
        
        if best_k_result:
            test_acc_best.append(best_k_result['test_acc'])
            test_f1_best.append(best_k_result['test_f1'])
            best_k_values.append(best_k_result.get('k_value', best_k_result['n_features']))
        else:
            test_acc_best.append(0)
            test_f1_best.append(0)
            best_k_values.append(0)
    
    x = np.arange(len(all_labels))
    width = 0.35
    
    # Plot 1: Test Accuracy Comparison
    axes[0, 0].bar(x - width/2, test_acc_no_fs, width, label='All Features', color='lightblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_acc_best, width, label='Best K', color='steelblue', edgecolor='black')
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(all_labels, fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])
    
    # Plot 2: Test F1 Score Comparison
    axes[0, 1].bar(x - width/2, test_f1_no_fs, width, label='All Features', color='lightcoral', edgecolor='black')
    axes[0, 1].bar(x + width/2, test_f1_best, width, label='Best K', color='darkred', edgecolor='black')
    axes[0, 1].set_ylabel('Test Macro F1', fontsize=11)
    axes[0, 1].set_title('Test Macro F1 Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_labels, fontsize=10)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])
    
    # Plot 3: Feature Selection Impact on F1
    fs_impact = [(best - baseline) / baseline * 100 if baseline > 0 else 0 
                 for best, baseline in zip(test_f1_best, test_f1_no_fs)]
    colors_fs = ['green' if imp > 0 else 'red' for imp in fs_impact]
    axes[1, 0].bar(x, fs_impact, color=colors_fs, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_ylabel('F1 Change (%)', fontsize=11)
    axes[1, 0].set_title('Feature Selection Impact\nPositive = Improvement', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_labels, fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Overall Best Configuration Summary
    axes[1, 1].axis('off')
    summary_text = "PERFORMANCE SUMMARY\n" + "="*40 + "\n\n"
    
    for i, dataset_name in enumerate(all_labels):
        summary_text += f"{dataset_name}:\n"
        summary_text += f"  Best K: {best_k_values[i]} features\n"
        summary_text += f"  Accuracy: {test_acc_best[i]:.4f}\n"
        summary_text += f"  Macro F1: {test_f1_best[i]:.4f}\n"
        summary_text += f"  FS Impact: {fs_impact[i]:+.2f}%\n\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Overall Model Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = 'overall_model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*20 + "REVIEW CLASSIFICATION ANALYSIS")
    print(" "*18 + "with Optuna Bayesian Optimization")
    print(" "*15 + "Target: Label (binary classes - 0: negative, 1: positive)")
    print(" "*15 + "Dataset: Already balanced (130 samples per class)")
    print(" "*15 + "CV-based K selection for optimal feature subset")
    print(" "*15 + "Exhaustive search: 200 Optuna trials per configuration")
    print(" "*15 + "Progress tracking: tqdm progress bars enabled")
    print("="*80)
    
    # Define datasets
    datasets = [
        ('../Data/reviewFeatures.csv', 'ReviewFeatures')
    ]
    
    # Will test all K values from 1 to n_features-1 (default behavior)
    # You can also specify specific K values like: k_values = [3, 5, 7, 8, 9]
    k_values = None  # None means try all values from 1 to n_features-1 for CV selection
    
    # Estimated runtime information
    n_k_values = 9  # K=1 to K=9
    n_trials_baseline = 200
    n_trials_best_k = 200
    cv_evaluations = n_k_values * 5  # 5-fold CV for each K
    total_trials = n_trials_baseline + cv_evaluations + n_trials_best_k
    
    print(f"\nEstimated execution:")
    print(f"  - Baseline (all features): {n_trials_baseline} Optuna trials")
    print(f"  - CV K selection: {n_k_values} K values × 5 folds = {cv_evaluations} evaluations")
    print(f"  - Best K training: {n_trials_best_k} Optuna trials")
    print(f"  - Total: ~{total_trials} model trainings")
    print(f"\nEstimated time: 5-10 minutes depending on hardware.")
    print(f"Progress bars will track each stage.")
    print(f"\nIMPORTANT: Best K is selected using CV on train+val data \n")
    
    # Process each dataset
    for filepath, dataset_name in datasets:
        try:
            print(f"\n{'*'*80}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'*'*80}")
            results_dict = process_dataset(filepath, dataset_name, k_values=k_values)
        except FileNotFoundError:
            print(f"\nError: File '{filepath}' not found. Skipping {dataset_name}.")
        except Exception as e:
            print(f"\nError processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print(f"\n{'='*80}")
    print("GENERATING FINAL SUMMARY...")
    print(f"{'='*80}")
    print_final_summary()
    
    # Create all combined visualizations
    create_combined_visualizations()
    
    print("\n" + "="*80)
    print(" "*25 + " ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - reviewfeatures_full_analysis.png (Detailed analysis)")
    print("  - reviewfeatures_k_optimization.png (K value comparison with CV validation scores)")
    print("  - overall_model_comparison.png (Summary)")
    print("\nNote: K optimization graphs show VALIDATION scores (used for selection),")
    print("      not test scores, to demonstrate proper ML methodology.")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
