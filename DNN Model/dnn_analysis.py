import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'k_features': 8,  # Number of features to select
    'n_splits': 5,    # Number of CV folds
    'seed': 42
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(filepath):
    """Load data from CSV."""
    return pd.read_csv(filepath)

def feature_selection(X, y, k=8):
    """
    Select top k features using Random Forest importance.
    Returns reduced X, selected feature names, and importance dataframe.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['seed'])
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    selected_indices = indices[:k]
    selected_features = X.columns[selected_indices]
    
    ranking_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return X[selected_features], selected_features, ranking_df

class BinaryClassifierDNN(nn.Module):
    """
    Deep Neural Network for binary classification.
    Includes Dropout and BatchNorm for regularization.
    """
    def __init__(self, input_dim):
        super(BinaryClassifierDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), np.array(all_labels), np.array(all_preds)

def main():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '../Data/reviewFeatures.csv')
    if not os.path.exists(data_path):
        data_path = "A:/Code/GitHub/CS273-HW2/Data/reviewFeatures.csv"
        
    df = load_data(data_path)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 2. Split Data (60/20/20 -> 80% Train+Val, 20% Test)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=CONFIG['seed']
    )
    
    # 3. Feature Selection (on Train+Val set to avoid leakage)
    X_tv_sel, selected_feats, feat_ranking = feature_selection(X_tv, y_tv, k=CONFIG['k_features'])
    X_test_sel = X_test[selected_feats]
    
    print(f"Selected Features: {list(selected_feats)}")
    
    # 4. Scaling
    scaler = StandardScaler()
    X_tv_scaled = scaler.fit_transform(X_tv_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Convert to Tensors
    X_tv_tensor = torch.FloatTensor(X_tv_scaled)
    y_tv_tensor = torch.FloatTensor(y_tv.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # 5. 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    
    best_f1 = 0.0
    best_model_state = None
    fold_results = []
    
    print("\nStarting Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tv_tensor, y_tv_tensor)):
        # Prepare Fold Data
        X_train_fold, X_val_fold = X_tv_tensor[train_idx], X_tv_tensor[val_idx]
        y_train_fold, y_val_fold = y_tv_tensor[train_idx], y_tv_tensor[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), 
                                  batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), 
                                batch_size=CONFIG['batch_size'], shuffle=False)
        
        # Initialize Model
        model = BinaryClassifierDNN(input_dim=len(selected_feats)).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
        
        # Training Loop
        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            
        # Validation
        val_loss, val_labels, val_preds = evaluate(model, val_loader, criterion)
        fold_acc = accuracy_score(val_labels, val_preds)
        fold_f1 = f1_score(val_labels, val_preds)
        
        print(f"Fold {fold+1}: Accuracy={fold_acc:.4f}, F1={fold_f1:.4f}")
        fold_results.append({'fold': fold+1, 'accuracy': fold_acc, 'f1': fold_f1})
        
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_model_state = model.state_dict()

    # 6. Save Model
    torch.save(best_model_state, "best_model_checkpoint.pth")
    print("\nBest model saved.")

    # 7. Final Evaluation on Test Set
    final_model = BinaryClassifierDNN(input_dim=len(selected_feats)).to(device)
    final_model.load_state_dict(best_model_state)
    
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=CONFIG['batch_size'])
    _, test_labels, test_preds = evaluate(final_model, test_loader, nn.BCELoss())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)
    
    print(f"\nTest Set Evaluation:\nAccuracy: {test_acc:.4f}\nF1 Score: {test_f1:.4f}")
    
    # 8. Plotting
    plt.figure(figsize=(14, 10))
    
    # Feature Importance
    plt.subplot(2, 2, 1)
    sns.barplot(x='Importance', y='Feature', data=feat_ranking.head(10), hue='Feature', palette='viridis', legend=False)
    plt.title('Top Feature Importances (RF)')
    
    # Confusion Matrix
    plt.subplot(2, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # CV Performance
    plt.subplot(2, 2, 3)
    folds = [r['fold'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    plt.plot(folds, accs, marker='o', label='Accuracy')
    plt.plot(folds, f1s, marker='s', label='F1 Score')
    plt.title('Cross-Validation Performance')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Summary Text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"Test Accuracy: {test_acc:.4f}\n"
        f"Test F1 Score: {test_f1:.4f}\n"
        f"Selected Features: {len(selected_feats)}\n"
        f"Best CV F1: {best_f1:.4f}\n"
        f"Device: {device}"
    )
    plt.text(0.1, 0.5, summary_text, fontsize=14)
    plt.title('Final Metrics')
    
    plt.tight_layout()
    plt.savefig('evaluation_graphs.png')
    plt.show()

if __name__ == "__main__":
    main()
