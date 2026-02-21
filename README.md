# CS273-HW2: Emotion Prediction from Acoustic Features

## Datasets
- **EmoSounds-3**: 600 samples, 75 acoustic features, bipolar scale (-1 to +1)
- **IADSED-2**: 935 samples, 76 acoustic features, Likert scale (1-9)

## Methods
- Preprocessing: Median imputation, Z-score standardization
- Feature engineering: Emotional intensity + quadrant features
- Split: 60% train / 20% validation / 20% test
- Models: Random Forest, XGBoost, DNN
- Balancing: SMOTE applied for comparison
- Feature selection: SelectKBest (k=34)

## Best Results

### EmoSounds-3
| Model | Configuration | Accuracy | Macro F1 |
|-------|--------------|----------|----------|
| **Random Forest** | Balanced + FS | 0.750 | **0.662** |
| XGBoost | Balanced + FS | 0.717 | 0.620 |
| DNN | Imbalanced + FS | 0.733 | 0.601 |

### IADSED-2
| Model | Configuration | Accuracy | Macro F1 |
|-------|--------------|----------|----------|
| **Random Forest** | Balanced + FS | 0.642 | **0.600** |
| XGBoost | Balanced + No FS | 0.636 | 0.587 |
| DNN | Imbalanced + FS | 0.599 | 0.551 |

## Key Findings
- **SMOTE balancing improved Macro F1 significantly** (>10% improvement in most cases)
- Feature selection provided modest gains when combined with balancing
- Random Forest performed best overall
- Negative arousal + negative valence classes were hardest to classify
- Class imbalance has greater impact than feature selection on performance

## Result Visualizations

### Random Forest Analysis
![EmoSounds-3 Random Forest Analysis](Random%20Forest%20Model/emosounds3_full_analysis.png)
![IADSED-2 Random Forest Analysis](Random%20Forest%20Model/iadsed2_full_analysis.png)
![Overall Model Comparison](Random%20Forest%20Model/overall_model_comparison.png)

### XGBoost Analysis
![EmoSounds-3 XGBoost Analysis](XGBoost/xgboost_emosounds3_full_analysis.png)
![IADSED-2 XGBoost Analysis](XGBoost/xgboost_iadsed2_full_analysis.png)
![XGBoost Overall Comparison](XGBoost/xgboost_overall_model_comparison.png)

### Deep Neural Network Analysis
![EmoSounds-3 DNN Confusion Matrix](Deep%20Neural%20Net%20Model/dnn_confusion_matrix_emosounds-3_imbalanced_-_feature_selection.png)
![EmoSounds-3 DNN Training History](Deep%20Neural%20Net%20Model/dnn_history_emosounds-3_imbalanced_-_feature_selection.png)
![IADSED-2 DNN Confusion Matrix](Deep%20Neural%20Net%20Model/dnn_confusion_matrix_iadsed-2_imbalanced_-_feature_selection.png)
![IADSED-2 DNN Training History](Deep%20Neural%20Net%20Model/dnn_history_iadsed-2_imbalanced_-_feature_selection.png)

### Data Visualizations
![EmoSounds-3 Data Visualizations](Data/visualizations_emosounds-3.png)
![IADSED-2 Data Visualizations](Data/visualizations_iadsed-2.png)