"""
CS273 Assignment Project 1 - Data Preprocessing

This script performs the following:
1. Load EmoSounds-3.csv and IADSED-2.csv datasets
2. Clean the datasets (handle NULL/N/A values)
3. Visualize the datasets and provide insights
4. Preprocess the datasets using standard techniques
5. Save preprocessed data as new CSV files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_datasets():
    """Load both datasets as Pandas DataFrames"""
    print("=" * 80)
    print("STEP 1: LOADING DATASETS")
    print("=" * 80)
    
    # Load EmoSounds dataset
    print("\nLoading EmoSounds-3.csv...")
    emosounds_df = pd.read_csv('EmoSounds-3.csv')
    print(f"EmoSounds dataset loaded: {emosounds_df.shape[0]} rows, {emosounds_df.shape[1]} columns")
    
    # Load IADSED dataset
    print("\nLoading IADSED-2.csv...")
    iadsed_df = pd.read_csv('IADSED-2.csv')
    print(f"IADSED dataset loaded: {iadsed_df.shape[0]} rows, {iadsed_df.shape[1]} columns")
    
    return emosounds_df, iadsed_df

def explore_data(df, dataset_name):
    """Explore and display basic information about the dataset"""
    print(f"\n{'=' * 80}")
    print(f"DATA EXPLORATION: {dataset_name}")
    print(f"{'=' * 80}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def clean_data(df, dataset_name):
    """Clean the dataset - handle NULL and N/A values"""
    print(f"\n{'=' * 80}")
    print(f"STEP 2: DATA CLEANING - {dataset_name}")
    print(f"{'=' * 80}")

    # Create a copy for cleaning
    df = df.copy()
    
    # Replace string representations of NaN with actual NaN
    nan_strings = ['nan', 'NaN', 'NAN', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 'None', '']
    df.replace(nan_strings, np.nan, inplace=True)
    
    # Check for missing values (including NaN)
    print(f"\nMissing values before cleaning:")
    missing_before = df.isnull().sum()
    missing_count = missing_before[missing_before > 0]
    
    if len(missing_count) > 0:
        print(missing_count)
        print(f"\nTotal missing values (including NaN): {df.isnull().sum().sum()}")
    else:
        print("No missing values found!")
    
    # Check for infinity values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_in_col = np.isinf(df[col]).sum()
        if inf_in_col > 0:
            print(f"  {col}: {inf_in_col} infinity values")
            inf_count += inf_in_col
            # Replace infinity with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    if inf_count > 0:
        print(f"\n Replaced {inf_count} infinity values with NaN")
    
    # Create a copy for cleaning
    df_cleaned = df.copy()
    
    # Separate numeric and non-numeric columns
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns
    
    print(f"\nNumeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    # Handle missing values in numeric columns
    if df_cleaned[numeric_columns].isnull().sum().sum() > 0:
        print(f"\nApplying imputation strategy for numeric columns...")
        print("Using median imputation for numeric features")
        
        # Use median imputation (more robust to outliers)
        imputer = SimpleImputer(strategy='median')
        df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])
        print("Numeric columns imputed with median values")
    
    # Handle missing values in categorical columns
    if len(categorical_columns) > 0 and df_cleaned[categorical_columns].isnull().sum().sum() > 0:
        print(f"\nApplying imputation strategy for categorical columns...")
        print("Using mode imputation for categorical features")
        
        for col in categorical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                df_cleaned[col].fillna(mode_value, inplace=True)
        print("Categorical columns imputed with mode values")
    
    # Remove duplicate rows
    duplicates_count = df_cleaned.duplicated().sum()
    if duplicates_count > 0:
        print(f"\nRemoving {duplicates_count} duplicate rows...")
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Duplicates removed")
    else:
        print(f"\nNo duplicate rows found")
    
    # Verify cleaning
    print(f"\nMissing values after cleaning:")
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Total missing values: {missing_after}")
    
    print(f"\n Data cleaning completed!")
    print(f"Final shape: {df_cleaned.shape}")
    
    return df_cleaned

def visualize_data(df, dataset_name):
    """Create visualizations and insights"""
    print(f"\n{'=' * 80}")
    print(f"STEP 3: DATA VISUALIZATION - {dataset_name}")
    print(f"{'=' * 80}")
    
    # Create a copy and replace string NaN values for accurate missing value detection
    df_viz = df.copy()
    nan_strings = ['nan', 'NaN', 'NAN', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 'None', '']
    df_viz.replace(nan_strings, np.nan, inplace=True)
    
    # Check if dominance exists to determine layout
    has_dominance = 'dominance' in df_viz.columns
    
    # Create a figure with multiple subplots
    if has_dominance:
        fig = plt.figure(figsize=(18, 14))
        rows, cols = 3, 3
    else:
        fig = plt.figure(figsize=(16, 10))
        rows, cols = 2, 3
    
    fig.suptitle(f'{dataset_name} - Data Analysis and Insights', fontsize=16, fontweight='bold')
    
    # Visualization 1: Distribution of emotional dimensions
    ax1 = plt.subplot(rows, cols, 1)
    if 'arousal' in df.columns and 'valence' in df.columns:
        # Detect scale type for threshold lines
        arousal_min = df['arousal'].min()
        valence_min = df['valence'].min()
        
        if arousal_min < 0 and valence_min < 0:
            arousal_threshold = 0
            valence_threshold = 0
        else:
            arousal_threshold = (df['arousal'].max() + df['arousal'].min()) / 2
            valence_threshold = (df['valence'].max() + df['valence'].min()) / 2
        
        ax1.scatter(df['valence'], df['arousal'], alpha=0.5, c='steelblue', edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Valence', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Arousal', fontsize=11, fontweight='bold')
        ax1.set_title('Arousal-Valence Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Draw threshold lines
        ax1.axhline(y=arousal_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=valence_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Visualization 2: Missing values heatmap
    ax2 = plt.subplot(rows, cols, 2)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:79]  # First 20 numeric columns
    missing_data = df[numeric_cols].isnull().sum()
    if missing_data.sum() > 0:
        missing_data[missing_data > 0].plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Missing Values per Column', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Columns', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='green')
        ax2.set_title('Missing Values Check', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Visualization 3: Distribution of key features
    ax3 = plt.subplot(rows, cols, 3)
    if 'arousal' in df.columns:
        ax3.hist(df['arousal'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Arousal', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Arousal Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Visualization 4: Correlation heatmap of top features
    ax4 = plt.subplot(rows, cols, 4)
    emotion_features = [col for col in ['arousal', 'valence', 'dominance'] if col in df.columns]
    acoustic_features = df.select_dtypes(include=[np.number]).columns[:7]
    features_to_correlate = sorted(list(set(emotion_features) | set(acoustic_features)))
    
    if len(features_to_correlate) > 1:
        corr_matrix = df[features_to_correlate].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Visualization 5: Valence distribution
    ax5 = plt.subplot(rows, cols, 5)
    if 'valence' in df.columns:
        ax5.hist(df['valence'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax5.set_xlabel('Valence', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Valence Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Visualization 6: Box plot of emotional dimensions
    ax6 = plt.subplot(rows, cols, 6)
    emotion_cols = [col for col in ['arousal', 'valence', 'dominance'] if col in df.columns]
    if emotion_cols:
        df[emotion_cols].boxplot(ax=ax6, patch_artist=True, 
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2))
        ax6.set_title('Emotional Dimensions - Box Plot', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Visualization 7: Dominance distribution (only if dominance exists)
    if has_dominance:
        ax7 = plt.subplot(rows, cols, 7)
        ax7.hist(df['dominance'], bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax7.set_xlabel('Dominance', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax7.set_title('Dominance Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Visualization 8: Dominance vs Arousal
        ax8 = plt.subplot(rows, cols, 8)
        if 'arousal' in df.columns:
            # Detect scale type for threshold lines
            dominance_min = df['dominance'].min()
            arousal_min = df['arousal'].min()
            
            if dominance_min < 0 and arousal_min < 0:
                dominance_threshold = 0
                arousal_threshold = 0
            else:
                dominance_threshold = (df['dominance'].max() + df['dominance'].min()) / 2
                arousal_threshold = (df['arousal'].max() + df['arousal'].min()) / 2
            
            ax8.scatter(df['dominance'], df['arousal'], alpha=0.5, c='mediumseagreen', edgecolors='black', linewidth=0.5)
            ax8.set_xlabel('Dominance', fontsize=11, fontweight='bold')
            ax8.set_ylabel('Arousal', fontsize=11, fontweight='bold')
            ax8.set_title('Dominance-Arousal Distribution', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            # Draw threshold lines
            ax8.axhline(y=arousal_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax8.axvline(x=dominance_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Visualization 9: Dominance vs Valence
        ax9 = plt.subplot(rows, cols, 9)
        if 'valence' in df.columns:
            # Detect scale type for threshold lines
            dominance_min = df['dominance'].min()
            valence_min = df['valence'].min()
            
            if dominance_min < 0 and valence_min < 0:
                dominance_threshold = 0
                valence_threshold = 0
            else:
                dominance_threshold = (df['dominance'].max() + df['dominance'].min()) / 2
                valence_threshold = (df['valence'].max() + df['valence'].min()) / 2
            
            ax9.scatter(df['dominance'], df['valence'], alpha=0.5, c='mediumpurple', edgecolors='black', linewidth=0.5)
            ax9.set_xlabel('Dominance', fontsize=11, fontweight='bold')
            ax9.set_ylabel('Valence', fontsize=11, fontweight='bold')
            ax9.set_title('Dominance-Valence Distribution', fontsize=12, fontweight='bold')
            ax9.grid(True, alpha=0.3)
            # Draw threshold lines
            ax9.axhline(y=valence_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax9.axvline(x=dominance_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save the visualization
    output_filename = f'visualizations_{dataset_name.replace(" ", "_").lower()}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n Visualization saved as: {output_filename}")
    
    # Print insights
    print(f"\n{'=' * 80}")
    print(f"KEY INSIGHTS - {dataset_name}")
    print(f"{'=' * 80}")
    
    if 'arousal' in df.columns and 'valence' in df.columns:
        print(f"\n1. EMOTIONAL DIMENSIONS:")
        print(f"Arousal range: [{df['arousal'].min():.3f}, {df['arousal'].max():.3f}]")
        print(f"Arousal mean: {df['arousal'].mean():.3f} (std: {df['arousal'].std():.3f})")
        print(f"Valence range: [{df['valence'].min():.3f}, {df['valence'].max():.3f}]")
        print(f"Valence mean: {df['valence'].mean():.3f} (std: {df['valence'].std():.3f})")
        
        # Detect scale type and calculate appropriate thresholds
        arousal_min = df['arousal'].min()
        valence_min = df['valence'].min()
        
        if arousal_min < 0 and valence_min < 0:
            # Bipolar scale centered at 0
            arousal_threshold = 0
            valence_threshold = 0
            scale_type = "bipolar (centered at 0)"
        else:
            # Likert scale - use midpoint
            arousal_threshold = (df['arousal'].max() + df['arousal'].min()) / 2
            valence_threshold = (df['valence'].max() + df['valence'].min()) / 2
            scale_type = f"Likert (midpoints: A={arousal_threshold:.2f}, V={valence_threshold:.2f})"
        
        # Emotional quadrants using appropriate thresholds
        high_arousal_pos_valence = ((df['arousal'] > arousal_threshold) & (df['valence'] > valence_threshold)).sum()
        high_arousal_neg_valence = ((df['arousal'] > arousal_threshold) & (df['valence'] < valence_threshold)).sum()
        low_arousal_pos_valence = ((df['arousal'] < arousal_threshold) & (df['valence'] > valence_threshold)).sum()
        low_arousal_neg_valence = ((df['arousal'] < arousal_threshold) & (df['valence'] < valence_threshold)).sum()
        
        print(f"\n2. EMOTIONAL QUADRANTS DISTRIBUTION ({scale_type}):")
        print(f"High Arousal + Positive Valence (Excited): {high_arousal_pos_valence} samples ({100*high_arousal_pos_valence/len(df):.1f}%)")
        print(f"High Arousal + Negative Valence (Angry/Tense): {high_arousal_neg_valence} samples ({100*high_arousal_neg_valence/len(df):.1f}%)")
        print(f"Low Arousal + Positive Valence (Calm/Relaxed): {low_arousal_pos_valence} samples ({100*low_arousal_pos_valence/len(df):.1f}%)")
        print(f"Low Arousal + Negative Valence (Sad/Bored): {low_arousal_neg_valence} samples ({100*low_arousal_neg_valence/len(df):.1f}%)")
    
    if 'dominance' in df.columns:
        print(f"\n3. DOMINANCE:")
        print(f"Range: [{df['dominance'].min():.3f}, {df['dominance'].max():.3f}]")
        print(f"Mean: {df['dominance'].mean():.3f} (std: {df['dominance'].std():.3f})")
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n4. DATA QUALITY:")
    print(f"Total features: {len(df.columns)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Outliers (values > 3 std): {sum((np.abs(df[numeric_cols] - df[numeric_cols].mean()) > 3 * df[numeric_cols].std()).sum())}")
    
    return fig

def preprocess_data(df, dataset_name):
    """Apply standard preprocessing techniques"""
    print(f"\n{'=' * 80}")
    print(f"STEP 4: DATA PREPROCESSING - {dataset_name}")
    print(f"{'=' * 80}")
    
    df_preprocessed = df.copy()
    
    # Separate numeric and non-numeric columns
    numeric_columns = df_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_preprocessed.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\nPreprocessing {len(numeric_columns)} numeric features...")
    
    # Save emotion labels separately (we don't want to scale target variables)
    emotion_labels = [col for col in ['arousal', 'valence', 'dominance'] if col in numeric_columns]
    feature_columns = [col for col in numeric_columns if col not in emotion_labels]
    
    print(f"Emotion labels (preserved): {len(emotion_labels)}")
    print(f"Feature columns (to be scaled): {len(feature_columns)}")
    
    # 1. Standardization (Z-score normalization)
    print(f"\n1. STANDARDIZATION (Z-score normalization):")
    print(f"Formula: z = (x - μ) / σ")
    
    scaler_standard = StandardScaler()
    df_preprocessed[feature_columns] = scaler_standard.fit_transform(df_preprocessed[feature_columns])
    print(f"Features standardized to mean=0, std=1")
    
    # 2. Handle outliers ( - using IQR method)
    print(f"\n2. OUTLIER DETECTION:")
    outlier_counts = {}
    for col in feature_columns[:10]:  # Check first 10 features
        Q1 = df_preprocessed[col].quantile(0.25)
        Q3 = df_preprocessed[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_preprocessed[col] < (Q1 - 3 * IQR)) | (df_preprocessed[col] > (Q3 + 3 * IQR))).sum()
        if outliers > 0:
            outlier_counts[col] = outliers
    
    if outlier_counts:
        print(f"Outliers detected in {len(outlier_counts)} features (using 3*IQR method)")
        for col, count in list(outlier_counts.items())[:5]:
            print(f"{col}: {count} outliers")
    else:
        print(f"No significant outliers detected (using 3*IQR method)")
    
    # 3. Feature engineering - create additional features
    print(f"\n3. FEATURE ENGINEERING:")
    
    if 'arousal' in df_preprocessed.columns and 'valence' in df_preprocessed.columns:
        # Emotional intensity (using original, non-standardized values)
        arousal_orig = df[['arousal']].values.flatten()
        valence_orig = df[['valence']].values.flatten()
        
        # Detect scale type: centered at 0 vs. positive range (Likert scale)
        arousal_min = arousal_orig.min()
        valence_min = valence_orig.min()
        
        # If both have negative values, scale is centered at 0
        # Otherwise, it's a Likert-style scale (use midpoint)
        if arousal_min < 0 and valence_min < 0:
            # Scale centered at 0 (e.g., [-1, 1])
            arousal_threshold = 0
            valence_threshold = 0
            scale_type = "bipolar (centered at 0)"
        else:
            # Likert scale (e.g., [1, 9]) - use midpoint
            arousal_threshold = (arousal_orig.max() + arousal_orig.min()) / 2
            valence_threshold = (valence_orig.max() + valence_orig.min()) / 2
            scale_type = f"Likert (thresholds: A={arousal_threshold:.2f}, V={valence_threshold:.2f})"
        
        print(f"Detected scale type: {scale_type}")
        
        # Calculate emotional intensity from original values
        df_preprocessed['emotional_intensity'] = np.sqrt(
            (arousal_orig - arousal_threshold)**2 + 
            (valence_orig - valence_threshold)**2
        )
        print(f"Created 'emotional_intensity' feature")
        
        # Emotional quadrant encoding using appropriate thresholds
        df_preprocessed['emotional_quadrant'] = 0
        df_preprocessed.loc[(arousal_orig > arousal_threshold) & (valence_orig > valence_threshold), 'emotional_quadrant'] = 1  # Excited
        df_preprocessed.loc[(arousal_orig > arousal_threshold) & (valence_orig < valence_threshold), 'emotional_quadrant'] = 2  # Angry
        df_preprocessed.loc[(arousal_orig < arousal_threshold) & (valence_orig < valence_threshold), 'emotional_quadrant'] = 3  # Sad
        df_preprocessed.loc[(arousal_orig < arousal_threshold) & (valence_orig > valence_threshold), 'emotional_quadrant'] = 4  # Calm
        
        # Count quadrant distribution
        quadrant_counts = df_preprocessed['emotional_quadrant'].value_counts().sort_index()
        print(f"Created 'emotional_quadrant' feature (1=Excited, 2=Angry, 3=Sad, 4=Calm)")
        print(f"Quadrant distribution:")
        for quad, count in quadrant_counts.items():
            if quad == 0:
                print(f"     - Neutral: {count} samples")
            elif quad == 1:
                print(f"     - Excited: {count} samples")
            elif quad == 2:
                print(f"     - Angry: {count} samples")
            elif quad == 3:
                print(f"     - Sad: {count} samples")
            elif quad == 4:
                print(f"     - Calm: {count} samples")
    
    # Summary statistics
    print(f"\n4. PREPROCESSING SUMMARY:")
    print(f"Original shape: {df.shape}")
    print(f"Preprocessed shape: {df_preprocessed.shape}")
    print(f"New features created: {df_preprocessed.shape[1] - df.shape[1]}")
    print(f"Scaling method: Z-score standardization")
    print(f"Missing values: {df_preprocessed.isnull().sum().sum()}")
    
    return df_preprocessed

def save_preprocessed_data(df, dataset_name):
    """Save preprocessed data to CSV"""
    print(f"\n{'=' * 80}")
    print(f"STEP 5: SAVING PREPROCESSED DATA - {dataset_name}")
    print(f"{'=' * 80}")
    
    output_filename = f'final_{dataset_name.replace(" ", "_").lower()}.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\n Preprocessed data saved as: {output_filename}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    return output_filename

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print(" " * 20 + "CS273 ASSIGNMENT PROJECT 1")
    print(" " * 25 + "DATA PREPROCESSING")
    print("=" * 80)
    
    # Step 1: Load datasets
    emosounds_df, iadsed_df = load_datasets()
    
    # Process EmoSounds dataset
    print("\n" + "#" * 80)
    print(" " * 25 + "PROCESSING EMOSOUNDS DATASET")
    print("#" * 80)
    
    visualize_data(emosounds_df, "EmoSounds-3")
    emosounds_cleaned = clean_data(emosounds_df, "EmoSounds-3")
    emosounds_preprocessed = preprocess_data(emosounds_cleaned, "EmoSounds-3")
    emo_output = save_preprocessed_data(emosounds_preprocessed, "EmoSounds-3")
    
    # Process IADSED dataset
    print("\n" + "#" * 80)
    print(" " * 25 + "PROCESSING IADSED DATASET")
    print("#" * 80)
    
    visualize_data(iadsed_df, "IADSED-2")
    iadsed_cleaned = clean_data(iadsed_df, "IADSED-2")
    iadsed_preprocessed = preprocess_data(iadsed_cleaned, "IADSED-2")
    iadsed_output = save_preprocessed_data(iadsed_preprocessed, "IADSED-2")
    
    print("\nPreprocessing Techniques Applied:")
    print("Missing value imputation (median for numeric, mode for categorical)")
    print("Duplicate removal")
    print("Z-score standardization")
    print("Outlier detection (IQR method)")
    print("Feature engineering (emotional intensity, emotional quadrants)")

if __name__ == "__main__":
    main()