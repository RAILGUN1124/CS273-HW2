import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imblearn not found. SMOTE balancing will be skipped.")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_preprocessed_datasets():
    """Load the preprocessed datasets"""
    print("=" * 80)
    print("STEP 1: LOADING PREPROCESSED DATASETS")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data')
    
    # Load EmoSounds dataset
    emo_path = os.path.join(data_dir, 'final_emosounds-3.csv')
    print(f"\nLoading {emo_path}...")
    if os.path.exists(emo_path):
        emosounds_df = pd.read_csv(emo_path)
        print(f"EmoSounds dataset loaded: {emosounds_df.shape[0]} rows, {emosounds_df.shape[1]} columns")
    else:
        print(f"Error: File not found at {emo_path}")
        emosounds_df = None
    
    # Load IADSED dataset
    iadsed_path = os.path.join(data_dir, 'final_iadsed-2.csv')
    print(f"\nLoading {iadsed_path}...")
    if os.path.exists(iadsed_path):
        iadsed_df = pd.read_csv(iadsed_path)
        print(f"IADSED dataset loaded: {iadsed_df.shape[0]} rows, {iadsed_df.shape[1]} columns")
    else:
        print(f"Error: File not found at {iadsed_path}")
        iadsed_df = None
    
    return emosounds_df, iadsed_df

class EmotionDNN(nn.Module):
    """Deep Neural Network for Emotion Classification"""
    def __init__(self, input_dim, num_classes):
        super(EmotionDNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.37)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.35)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3)
        )
        
        self.output_layer = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=25):
    """Train the PyTorch model with Early Stopping"""
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        scheduler.step(val_epoch_loss)
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def perform_dnn_classification(df, dataset_name):
    """
    Train and evaluate a 4-class DNN classifier using PyTorch.
    """
    print(f"\n{'=' * 80}")
    print(f"STEP 2: DNN CLASSIFICATION & EVALUATION (PyTorch) - {dataset_name}")
    print(f"{'=' * 80}")
    
    if df is None:
        print("Dataset is None, skipping...")
        return
    
    # 1. Prepare Data
    if 'emotional_quadrant' not in df.columns:
        print("Error: 'emotional_quadrant' column not found. Please run preprocessing first.")
        return
    
    df_model = df[df['emotional_quadrant'] != 0].copy()
    
    y = df_model['emotional_quadrant']
    
    exclude_cols = ['emotional_quadrant', 'arousal', 'valence', 'dominance', 'emotional_intensity']
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df_model[feature_cols]
    
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Class distribution:\n{y.value_counts(normalize=True).sort_index()}")
    
    # Encode labels to 0-3 range for PyTorch
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    # 2. Split Data (80% Train+Val, 20% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_val = pd.DataFrame(scaler.fit_transform(X_train_val), columns=X_train_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"Train+Val set: {X_train_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    results = []
    
    scenarios = [
        {'name': 'Imbalanced - All Features', 'balance': False, 'select_features': False},
        {'name': 'Imbalanced - Feature Selection', 'balance': False, 'select_features': True},
        {'name': 'Balanced - All Features', 'balance': True, 'select_features': False},
        {'name': 'Balanced - Feature Selection', 'balance': True, 'select_features': True},
    ]
    
    for scenario in scenarios:
        print(f"\nRunning Scenario: {scenario['name']}...")
        
        # Copy data for this scenario
        X_curr_train = X_train_val.copy()
        y_curr_train = y_train_val.copy()
        X_curr_test = X_test.copy()
        
        # Feature Selection
        if scenario['select_features']:
            k = min(20, X_curr_train.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            X_curr_train = selector.fit_transform(X_curr_train, y_curr_train)
            X_curr_test = selector.transform(X_curr_test)
            print(f"  Selected top {k} features")
        else:
            X_curr_train = X_curr_train.values
            X_curr_test = X_curr_test.values
        
        # Balancing (SMOTE) - applied only to training data
        if scenario['balance'] and HAS_IMBLEARN:
            smote = SMOTE(random_state=42)
            X_curr_train, y_curr_train = smote.fit_resample(X_curr_train, y_curr_train)
            print(f"  Applied SMOTE balancing. New training size: {X_curr_train.shape[0]}")
        
        # Cross-Validation for Hyperparameter Tuning / Model Selection
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        print("  Performing 5-fold Cross-Validation...")
        
        fold_idx = 1
        for train_idx, val_idx in kfold.split(X_curr_train, y_curr_train):
            # Split for this fold
            X_fold_train, X_fold_val = X_curr_train[train_idx], X_curr_train[val_idx]
            y_fold_train, y_fold_val = y_curr_train[train_idx], y_curr_train[val_idx]
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_fold_train), 
                torch.LongTensor(y_fold_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_fold_val), 
                torch.LongTensor(y_fold_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Create model
            model = EmotionDNN(X_fold_train.shape[1], num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) # weight_decay for L2 regularization
            
            # Train
            _ = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10)
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total
            cv_scores.append(acc)
            fold_idx += 1
            
        avg_cv_acc = np.mean(cv_scores)
        print(f"  Average CV Accuracy: {avg_cv_acc:.4f}")
        
        print("  Training final model on full training set...")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_curr_train), 
            torch.LongTensor(y_curr_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_curr_test), 
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        final_model = EmotionDNN(X_curr_train.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        history = train_model(final_model, train_loader, test_loader, criterion, optimizer, num_epochs=100, patience=25)
        
        final_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = final_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test Macro F1: {f1:.4f}")
        
        results.append({
            'Scenario': scenario['name'],
            'Accuracy': acc,
            'Macro F1': f1,
            'CV Accuracy': avg_cv_acc
        })
        
        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        
        # Labels: 1, 2, 3, 4
        label_names = ['PA-PV (Excited)', 'PA-NV (Angry)', 'NA-PV (Calm)', 'NA-NV (Sad)']
        
        # Ensure we only use labels that exist in the data if some classes are missing
        unique_labels = sorted(list(set(all_labels) | set(all_preds)))
        present_label_names = [label_names[i] for i in unique_labels]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=present_label_names,
                    yticklabels=present_label_names)
        plt.title(f'DNN Confusion Matrix (Hallucination Matrix)\n{dataset_name} - {scenario["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save in current directory
        filename = f'dnn_confusion_matrix_{dataset_name.replace(" ", "_").lower()}_{scenario["name"].replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        plt.close()
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Test')
        plt.title('DNN Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Test')
        plt.title('DNN Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        hist_filename = f'dnn_history_{dataset_name.replace(" ", "_").lower()}_{scenario["name"].replace(" ", "_").lower()}.png'
        plt.savefig(hist_filename)
        plt.close()
        
    # Display Comparison
    results_df = pd.DataFrame(results)
    print(f"\nDNN Results Comparison - {dataset_name}:")
    print(results_df[['Scenario', 'Accuracy', 'Macro F1', 'CV Accuracy']])
    
    # Save results to CSV
    results_df.to_csv(f'dnn_classification_results_{dataset_name.replace(" ", "_").lower()}.csv', index=False)
    
    return results_df

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print(" " * 20 + "CS273 ASSIGNMENT PROJECT 1")
    print(" " * 25 + "DEEP NEURAL NETWORK CLASSIFICATION (PyTorch)")
    print("=" * 80)
    
    # Step 1: Load preprocessed datasets
    emosounds_df, iadsed_df = load_preprocessed_datasets()
    
    # Step 2: Perform Classification
    if emosounds_df is not None:
        perform_dnn_classification(emosounds_df, "EmoSounds-3")
    
    if iadsed_df is not None:
        perform_dnn_classification(iadsed_df, "IADSED-2")
    
    print("\nDNN Classification Completed!")

if __name__ == "__main__":
    main()
