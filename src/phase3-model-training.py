import numpy as np
import pandas as pd
import sys
import ast # For safely evaluating string representations of lists
from pathlib import Path
from collections import defaultdict

from keras.src.optimizers import Adam
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
# Scikit-learn imports
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# TensorFlow/Keras imports
# Make TensorFlow logs less verbose
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

from xgboost import XGBClassifier

# --- Constants ---
N_SPLITS = 5
RANDOM_STATE = 42 # for reproducibility
SEMANTIC_DIM = 384 # Dimension of 'all-MiniLM-L6-v2' embeddings
EPOCHS = 50 # Max epochs for Keras models
BATCH_SIZE = 32
PATIENCE = 5 # Early stopping patience
#
# # Model candidates
# models = {
#     'logistic_regression': LogisticRegression(
#         penalty='l2',
#         C=1.0,
#         solver='liblinear',
#         class_weight='balanced'
#     ),
#     'linear_svm': LinearSVC(
#         class_weight='balanced',
#         dual=False,  # Recommended for n_samples > n_features
#         max_iter=10000
#     ),
#     'random_forest': RandomForestClassifier(
#         n_estimators=100,
#         max_depth=None,
#         class_weight='balanced_subsample'
#     ),
#     'naive_bayes': ComplementNB(),
#     'xgb': XGBClassifier(
#         tree_method='hist',
#         enable_categorical=True,
#         #scale_pos_weight=ratio_negative/ratio_positive # Needs ratio_negative, ratio_positive to be defined.  Calculate this from y_train
#     )
# }
# --- Model Building Functions ---


def build_svm():
    """Builds a Support Vector Classifier."""
    # Added probability=True for potential later use (e.g., ensembling)
    return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)

def build_cnn(input_dim):
    """Builds a 1D CNN model for flat feature vectors (e.g., syntactic)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        # Reshape flat vector into (timesteps, features_per_step) = (input_dim, 1) for Conv1D
        Reshape((input_dim, 1)),
        Conv1D(128, 5, activation='relu'), # Increased filters, larger kernel
        GlobalMaxPooling1D(),
        Dropout(0.5), # Added dropout for regularization
        Dense(64, activation='relu'),
        Dropout(0.5), # Added dropout
        Dense(1, activation='sigmoid') # Binary classification output
    ], name="CNN_Syntactic_Model")
    model.compile(optimizer=Adam(learning_rate=0.0005), # Adjusted learning rate
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_dense_for_semantic(input_dim):
    """Builds a simple Dense network for semantic vectors."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.5), # Added dropout
        Dense(64, activation='relu'),
        Dropout(0.5), # Added dropout
        Dense(1, activation='sigmoid') # Binary classification output
    ], name="Dense_Semantic_Model")
    model.compile(optimizer=Adam(learning_rate=0.0005), # Adjusted learning rate
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# --- Data Loading and Preprocessing ---

def load_and_preprocess_data(file_path: Path):
    """Loads data, parses vectors, encodes labels, and handles missing columns."""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # 1. Check for essential columns
    required_cols = ['clean_text', 'label']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Input CSV missing required columns: {missing}")
        sys.exit(1)

    # 2. Parse 'semantic_vector' if it exists
    if 'semantic_vector' in df.columns:
        print("Parsing semantic vectors...")
        try:
            # Use apply with a lambda function and ast.literal_eval
            # Ensure robustness against potential non-string or malformed entries
            def safe_literal_eval(x):
                try:
                    return np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array([])
                except (ValueError, SyntaxError):
                    return np.array([]) # Return empty array on error

            df['semantic_vector'] = df['semantic_vector'].apply(safe_literal_eval)

            # Check for empty arrays resulting from parsing errors
            empty_vectors = df['semantic_vector'].apply(lambda x: x.size == 0)
            if empty_vectors.any():
                print(f"Warning: Found {empty_vectors.sum()} rows with invalid semantic vectors. These rows might cause issues.")
                # Consider dropping these rows: df = df[~empty_vectors]

            # Verify the dimension of the first valid vector
            first_valid_vector = df.loc[~empty_vectors, 'semantic_vector'].iloc[0]
            if first_valid_vector.size > 0 and first_valid_vector.shape[0] != SEMANTIC_DIM:
                print(f"Warning: Expected semantic vector dimension {SEMANTIC_DIM}, but found {first_valid_vector.shape[0]}. Adjust SEMANTIC_DIM if needed.")
                # Optionally, exit or try to handle the mismatch
                # sys.exit(1)
            elif first_valid_vector.size == 0 and not empty_vectors.all():
                print("Warning: Could not verify semantic vector dimension due to parsing issues at the beginning.")

        except Exception as e:
            print(f"An unexpected error occurred during semantic vector parsing: {e}")
            # Depending on severity, you might want to exit
            # sys.exit(1)
    else:
        print("Warning: 'semantic_vector' column not found. Semantic features will be skipped.")


    # 3. Encode 'label' column
    print("Encoding labels...")
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    print(f"Labels mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    label_classes = le.classes_

    # 4. Handle potentially missing 'profanity_score'
    if 'profanity_score' not in df.columns:
        print("Warning: 'profanity_score' column not found. Setting to 0.0 for lexical features.")
        df['profanity_score'] = 0.0 # Add a default value

    # 5. Handle TF-IDF features (check for dense columns, assume sparse is separate)
    tfidf_sparse_path = Path('data/processed/tfidf_matrix.npz')
    tfidf_cols_dense = [col for col in df.columns if col.startswith('tfidf_')]

    if not tfidf_sparse_path.exists():
        print(f"Warning: Sparse TF-IDF matrix file not found at {tfidf_sparse_path}. 'tfidf_sparse' group will be skipped.")

    if not tfidf_cols_dense:
        print("Warning: No dense TF-IDF columns (starting with 'tfidf_') found. 'tfidf_dense' group will be skipped.")
    else:
        print(f"Found {len(tfidf_cols_dense)} dense TF-IDF columns ('tfidf_*'). Converting to numeric.")
        for col in tfidf_cols_dense:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        print(f"Warning: Column {col} contained non-numeric values converted to NaN. Filling with 0.")
                        df[col].fillna(0, inplace=True) # Simple imputation
                except Exception as e:
                    print(f"Error converting column {col} to numeric: {e}. Check data quality.")
                    # Consider dropping the column or exiting
                    # df = df.drop(columns=[col])
                    # sys.exit(1)

    # Remove potentially problematic list-based tfidf columns from previous runs if they exist
    for col_name in ['tfidf', 'tfidf_feature']:
        if col_name in df.columns:
            print(f"Removing potentially problematic column '{col_name}'.")
            df = df.drop(columns=[col_name])

    # Check for NaNs in critical columns after processing
    df.dropna(subset=['clean_text', 'label_encoded'], inplace=True)
    print(f"Data shape after initial processing and NaN drop: {df.shape}")

    return df, label_classes

# --- Model Training and Evaluation ---

# --- Model Training and Evaluation ---

def train_evaluate_models(df, label_classes):
    """Trains and evaluates models for different feature groups using cross-validation."""

    # Define feature groups, checking for column existence
    feature_groups = {
        'lexical': [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df.columns],
        'syntactic': [col for col in df if col.startswith('pos_')],
        # Only include semantic if column exists and has been processed
        'semantic': ['semantic_vector'] if 'semantic_vector' in df.columns and df['semantic_vector'].iloc[0].size > 0 else [],
        # Use filename as placeholder for sparse tfidf, load later
        'tfidf_sparse': ['tfidf_matrix.npz'] if Path('data/processed/tfidf_matrix.npz').exists() else [],
        # Only include dense tfidf if columns exist
        'tfidf_dense': [col for col in df if col.startswith('tfidf_')],
    }

    # Filter out empty feature groups
    feature_groups = {name: cols for name, cols in feature_groups.items() if cols}

    print("\nFeature groups to be evaluated:")
    for name, cols in feature_groups.items():
        if name == 'tfidf_sparse':
            print(f"- {name}: Features will be loaded from {cols[0]}")
        elif name == 'semantic':
            print(f"- {name}: {SEMANTIC_DIM} features from '{cols[0]}'")
        else:
            print(f"- {name}: {len(cols)} features: {cols[:5]}..." if len(cols) > 5 else cols) # Print first few column names

    # Target variable
    y = df['label_encoded'].values

    # Cross-Validation setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = [] # To store results from each fold, model, and group

    # Keras early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)

    # --- Iterate through defined feature groups ---
    for group_name, feature_cols in feature_groups.items():
        print(f"\n{'='*10} Training models for feature group: {group_name} {'='*10}")

        is_keras_model_group = group_name in ['syntactic', 'semantic']
        is_sklearn_group = group_name in ['lexical', 'tfidf_sparse', 'tfidf_dense']

        # --- Prepare data (X) for the current group ---
        X = None
        input_dim = 0
        is_sparse = False

        try:
            if group_name == 'semantic':
                # Stack the vectors into a 2D numpy array
                X = np.stack(df[feature_cols[0]].values)
                input_dim = X.shape[1]
                if input_dim != SEMANTIC_DIM:
                    print(f"  Info: Actual semantic vector dimension ({input_dim}) differs from expected ({SEMANTIC_DIM}). Using actual dimension.")
            elif group_name == 'tfidf_sparse':
                tfidf_path = Path('data/processed') / feature_cols[0]
                X = load_npz(tfidf_path)
                # Ensure X has the same number of rows as y
                if X.shape[0] != len(y):
                    print(f"  Error: Row mismatch between TF-IDF matrix ({X.shape[0]}) and labels ({len(y)}). Check data consistency.")
                    continue # Skip this group
                input_dim = X.shape[1]
                is_sparse = True
                print(f"  Loaded sparse TF-IDF matrix with shape: {X.shape}")
            elif group_name in ['lexical', 'syntactic', 'tfidf_dense']:
                X = df[feature_cols].values
                # Handle potential NaNs again (e.g., if imputation failed earlier)
                if np.isnan(X).any():
                    print(f"  Warning: NaN values found in feature group '{group_name}'. Imputing with 0.")
                    X = np.nan_to_num(X) # Replace NaN with 0, Inf with large numbers
                input_dim = X.shape[1]
            else:
                print(f"  Skipping unrecognized feature group type: {group_name}")
                continue

            if X is None or input_dim == 0:
                print(f"  Skipping group '{group_name}' due to data preparation issues or zero features.")
                continue
            if X.shape[0] != len(y):
                print(f"  Error: Feature matrix shape {X.shape} doesn't match label length {len(y)} for group '{group_name}'. Skipping.")
                continue

        except Exception as e:
            print(f"  Error preparing data for group '{group_name}': {e}")
            continue # Skip this group if data prep fails

        # --- Cross-Validation Loop ---
        fold_results_group = defaultdict(lambda: defaultdict(list)) # Store metrics: fold_results_group[model_name][metric_name] = [values]

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n  --- Fold {fold+1}/{N_SPLITS} ---")

            # --- Data Splitting ---
            if is_sparse:
                X_train, X_val = X[train_idx], X[val_idx]
            else:
                X_train, X_val = X[train_idx].astype(np.float32), X[val_idx].astype(np.float32) # Ensure float32 for scaling/Keras
            y_train, y_val = y[train_idx], y[val_idx]

            # --- Scaling (Fit on Train, Transform Train & Val) ---
            if not is_sparse:
                scaler = StandardScaler()
                try:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    print(f"    Data scaled (StandardScaler).")
                except ValueError as e:
                    print(f"    Warning: Scaling failed for fold {fold+1}: {e}. Using unscaled data.")
                    X_train_scaled = X_train
                    X_val_scaled = X_val
            else:
                X_train_scaled = X_train # Use original sparse matrix
                X_val_scaled = X_val
                print(f"    Using sparse data (no scaling applied).")

            # --- Define Models for this Fold ---
            current_models_to_run = {}
            fold_random_state = RANDOM_STATE + fold # Consistent random state per fold

            if is_sklearn_group:
                # Calculate scale_pos_weight for XGBoost based on training fold
                count_neg = np.sum(y_train == 0)
                count_pos = np.sum(y_train == 1)
                scale_pos_weight_val = count_neg / count_pos if count_pos > 0 else 1

                sklearn_models = {
                    'LogisticRegression': LogisticRegression(
                        penalty='l2', C=1.0, solver='liblinear', class_weight='balanced',
                        random_state=fold_random_state, max_iter=2000 # Increased max_iter
                    ),
                    'LinearSVC': LinearSVC(
                        class_weight='balanced', dual='auto', # Let sklearn choose based on n_samples/n_features
                        random_state=fold_random_state, max_iter=5000, C=1.0, tol=1e-3 # Relax tolerance slightly
                    ),
                    'RandomForest': RandomForestClassifier(
                        n_estimators=150, max_depth=None, class_weight='balanced_subsample',
                        random_state=fold_random_state, n_jobs=-1, min_samples_split=5 # Added min_samples_split
                    ),
                    'ComplementNB': ComplementNB(),
                    'XGBoost': XGBClassifier(
                        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                        tree_method='hist',
                        scale_pos_weight=scale_pos_weight_val,
                        random_state=fold_random_state,
                        n_estimators=100
                    )
                }
                # Add RBF SVM only for 'lexical' group as it's slow on high dimensions
                if group_name == 'lexical':
                    sklearn_models['SVC_RBF'] = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=fold_random_state)

                current_models_to_run.update(sklearn_models)

            elif is_keras_model_group:
                tf.random.set_seed(fold_random_state) # Seed for reproducibility per fold
                if group_name == 'syntactic':
                    model = build_cnn(input_dim=input_dim)
                    current_models_to_run[model.name] = model
                elif group_name == 'semantic':
                    model = build_dense_for_semantic(input_dim=input_dim)
                    current_models_to_run[model.name] = model

            # --- Train and Evaluate each model for the current fold ---
            for model_name, model_instance in current_models_to_run.items():
                print(f"    Processing Model: {model_name}")

                # Prepare data specific to model requirements (dense/sparse)
                current_X_train = X_train_scaled
                current_X_val = X_val_scaled
                model_requires_dense = model_name in ['RandomForest', 'XGBoost', 'SVC_RBF'] or is_keras_model_group

                if is_sparse and model_requires_dense:
                    print(f"      Converting data to dense for {model_name}...")
                    try:
                        current_X_train = current_X_train.toarray()
                        current_X_val = current_X_val.toarray()
                    except MemoryError:
                        print(f"      MemoryError converting sparse data to dense for {model_name}. Skipping model.")
                        continue
                    except Exception as e:
                        print(f"      Error converting data to dense for {model_name}: {e}. Skipping model.")
                        continue

                # --- Training ---
                try:
                    if is_keras_model_group and model_name == model_instance.name: # Check if it's the Keras model
                        print(f"      Training {model_name}...")
                        history = model_instance.fit(current_X_train, y_train,
                                                     epochs=EPOCHS,
                                                     batch_size=BATCH_SIZE,
                                                     validation_data=(current_X_val, y_val),
                                                     callbacks=[early_stopping],
                                                     verbose=0) # Keep verbose=0 for cleaner logs during CV
                        print(f"      Training completed after {len(history.history['loss'])} epochs.")
                        # --- Evaluation (Keras) ---
                        y_pred_proba = model_instance.predict(current_X_val, batch_size=BATCH_SIZE*2).flatten()
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        tf.keras.backend.clear_session() # Clear session to free memory
                        del model_instance # Explicitly delete model
                        import gc
                        gc.collect() # Force garbage collection

                    else: # Scikit-learn models
                        print(f"      Training {model_name}...")
                        model_instance.fit(current_X_train, y_train)
                        print(f"      Training completed.")
                        # --- Evaluation (Scikit-learn) ---
                        y_pred = model_instance.predict(current_X_val)

                except Exception as e:
                    print(f"      ERROR during training/prediction for {model_name}: {e}")
                    # Assign default bad scores or skip metrics
                    accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan # Use NaN for errors
                    # Continue to the next model, but record the failure
                    fold_results_group[model_name]['fold'].append(fold + 1)
                    fold_results_group[model_name]['accuracy'].append(accuracy)
                    fold_results_group[model_name]['precision'].append(precision)
                    fold_results_group[model_name]['recall'].append(recall)
                    fold_results_group[model_name]['f1_score'].append(f1)
                    print(f"      Fold {fold+1} Metrics ({model_name}): FAILED")
                    continue # Skip to next model

                # --- Calculate Metrics ---
                accuracy = accuracy_score(y_val, y_pred)
                # Use binary average, handle potential division by zero in precision/recall
                precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)

                # Store results for this model and fold
                fold_results_group[model_name]['fold'].append(fold + 1)
                fold_results_group[model_name]['accuracy'].append(accuracy)
                fold_results_group[model_name]['precision'].append(precision)
                fold_results_group[model_name]['recall'].append(recall)
                fold_results_group[model_name]['f1_score'].append(f1)
                print(f"      Fold {fold+1} Metrics ({model_name}): Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                # Optional: Detailed report per fold
                # print(classification_report(y_val, y_pred, target_names=[str(c) for c in label_classes], zero_division=0))


        # --- Aggregate results for the feature group (after all folds) ---
        print(f"\n  --- Aggregated Results for Feature Group: {group_name} ---")
        for model_name, metrics_dict in fold_results_group.items():
            # Use np.nanmean and np.nanstd to ignore NaNs from failed folds/models
            avg_accuracy = np.nanmean(metrics_dict['accuracy'])
            avg_precision = np.nanmean(metrics_dict['precision'])
            avg_recall = np.nanmean(metrics_dict['recall'])
            avg_f1 = np.nanmean(metrics_dict['f1_score'])

            std_accuracy = np.nanstd(metrics_dict['accuracy'])
            std_precision = np.nanstd(metrics_dict['precision'])
            std_recall = np.nanstd(metrics_dict['recall'])
            std_f1 = np.nanstd(metrics_dict['f1_score'])

            num_successful_folds = len(metrics_dict['fold']) - np.isnan(metrics_dict['accuracy']).sum()

            print(f"    Model: {model_name} ({num_successful_folds}/{N_SPLITS} successful folds)")
            if num_successful_folds > 0:
                print(f"      Avg Accuracy:  {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
                print(f"      Avg Precision: {avg_precision:.4f} (+/- {std_precision:.4f})")
                print(f"      Avg Recall:    {avg_recall:.4f} (+/- {std_recall:.4f})")
                print(f"      Avg F1-Score:  {avg_f1:.4f} (+/- {std_f1:.4f})")
            else:
                print("      All folds failed for this model.")

            # Append all fold results (including NaNs for failures) to the main results list
            for i in range(len(metrics_dict['fold'])):
                all_results.append({
                    'feature_group': group_name,
                    'model_name': model_name,
                    'fold': metrics_dict['fold'][i],
                    'accuracy': metrics_dict['accuracy'][i],
                    'precision': metrics_dict['precision'][i],
                    'recall': metrics_dict['recall'][i],
                    'f1_score': metrics_dict['f1_score'][i]
                })

    # --- Final Processing ---
    results_df = pd.DataFrame(all_results)
    return results_df

# def train_tfidf_models(X, y):
#     results = {}
#
#     for name, model in models.items():
#         # Sparse matrix handling
#         if name in ['logistic_regression', 'linear_svm', 'naive_bayes']:
#             X_data = X
#         else:  # Convert to dense for tree-based models
#             X_data = X.toarray()
#
#         # Cross-validated predictions
#         y_pred = cross_val_predict(model, X_data, y, cv=5, n_jobs=-1)
#
#         # Store results
#         results[name] = {
#             'report': classification_report(y, y_pred, output_dict=True),
#             'features_used': X.shape[1]
#         }
#
#     return results

# --- Main Execution ---
if __name__ == "__main__":
    # Define paths
    data_dir = Path('data/processed')
    # Input file should be the output of phase 2 (feature engineering)
    input_file = data_dir / 'phase2_output.csv' # Make sure this file exists and has the expected columns
    output_dir = Path('results/phase3') # Specific output dir for this phase
    output_file = output_dir / 'model_training_results.csv'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df, label_classes = load_and_preprocess_data(input_file)

    # Check if DataFrame is empty after preprocessing
    if df.empty:
        print("Error: DataFrame is empty after preprocessing. Cannot proceed with training.")
        sys.exit(1)

    # Train and evaluate models
    results_df = train_evaluate_models(df, label_classes)

    # Save results
    if not results_df.empty:
        print(f"\nSaving detailed results to: {output_file}")
        results_df.to_csv(output_file, index=False)

        # Display average results per group and model, handling potential NaNs
        print("\n--- Average Performance Across Folds (Grouped by Feature Group & Model) ---")
        # Use pivot_table for a potentially cleaner view
        try:
            avg_results_pivot = pd.pivot_table(results_df,
                                               index=['feature_group', 'model_name'],
                                               values=['accuracy', 'precision', 'recall', 'f1_score'],
                                               aggfunc=[np.nanmean, np.nanstd])
            print(avg_results_pivot)
        except Exception as e:
            print(f"Could not generate pivot table summary: {e}")
            print("Displaying basic grouped summary:")
            avg_results = results_df.groupby(['feature_group', 'model_name'])[['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean', 'std'])
            print(avg_results)

    else:
        print("\nNo results were generated during model training and evaluation.")

    print("\nProcessing complete.")