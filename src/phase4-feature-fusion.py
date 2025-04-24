import joblib
import numpy as np
import pandas as pd
import sys
import ast  # For safely evaluating string representations of lists
from pathlib import Path
from collections import defaultdict
import gc  # Garbage collector

from scipy.sparse import load_npz, hstack as sparse_hstack

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# TensorFlow/Keras imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow verbosity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalMaxPooling1D, Dropout, Bidirectional, concatenate, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- Constants ---
N_SPLITS = 5 # Number of folds for cross-validation on the training set
RANDOM_STATE = 42
SEMANTIC_DIM = 384 # Dimension of 'all-MiniLM-L6-v2' embeddings
EPOCHS = 50 # Max epochs for Keras models
BATCH_SIZE = 32
PATIENCE = 5 # Early stopping patience
TEST_SET_SIZE = 0.2 # Proportion of data to hold out for the final test set

# --- Model Building Functions (Keep as before) ---

def build_svm(kernel='linear', probability=False, class_weight='balanced', random_state=None):
    """Builds a Support Vector Classifier."""
    if kernel == 'linear':
        base_svm = LinearSVC(
            class_weight=class_weight,
            dual='auto',
            random_state=random_state,
            max_iter=5000,
            C=1.0,
            tol=1e-3
        )
        if probability:
            return CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
        else:
            return base_svm
    else:
        return SVC(kernel=kernel, probability=probability, class_weight=class_weight, random_state=random_state)


def build_dense_for_semantic(input_dim):
    """Builds a simple Dense network for semantic vectors."""
    model = Sequential([
        Input(shape=(input_dim,), name="semantic_input"),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="Dense_Semantic_Model")
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_bilstm_intermediate(semantic_dim, syntactic_dim):
    """Builds a Bi-LSTM model for intermediate fusion of semantic and syntactic features."""
    semantic_input = Input(shape=(semantic_dim,), name='semantic_input')
    syntactic_input = Input(shape=(syntactic_dim,), name='syntactic_input')
    semantic_reshaped = tf.keras.layers.Reshape((1, semantic_dim))(semantic_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(semantic_reshaped)
    combined = concatenate([lstm_out, syntactic_input])
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[semantic_input, syntactic_input], outputs=output, name="BiLSTM_Intermediate_Fusion")
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Data Loading and Preprocessing (Keep as before) ---

def safe_literal_eval(x):
    """Safely evaluates a string literal, returning an empty list on error."""
    try:
        if isinstance(x, str):
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.float32)
        return np.array([], dtype=np.float32)
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)

def load_and_preprocess_data(file_path: Path, tfidf_sparse_path: Path):
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

    # 2. Parse 'semantic_vector'
    if 'semantic_vector' in df.columns:
        print("Parsing semantic vectors...")
        df['semantic_vector'] = df['semantic_vector'].apply(safe_literal_eval)
        empty_vectors = df['semantic_vector'].apply(lambda x: x.size == 0)
        if empty_vectors.any():
            print(f"Dropping {empty_vectors.sum()} rows with invalid semantic vectors.")
            df = df[~empty_vectors].copy()
            df.reset_index(drop=True, inplace=True)
            if df.empty:
                print("Error: DataFrame is empty after removing rows with invalid semantic vectors.")
                sys.exit(1)
        if not df.empty:
            first_valid_vector_dim = df['semantic_vector'].iloc[0].shape[0]
            # Use the actual dimension found in the data if it differs from the constant
            global SEMANTIC_DIM # Allow modification of the global constant
            if first_valid_vector_dim != SEMANTIC_DIM:
                print(f"Warning: Expected semantic vector dimension {SEMANTIC_DIM}, but found {first_valid_vector_dim}. Using actual dimension: {first_valid_vector_dim}")
                SEMANTIC_DIM = first_valid_vector_dim # Update global constant
            inconsistent_dims = df['semantic_vector'].apply(lambda x: x.shape[0] != SEMANTIC_DIM).sum()
            if inconsistent_dims > 0:
                print(f"Error: Found {inconsistent_dims} semantic vectors with inconsistent dimensions.")
                sys.exit(1)
    else:
        print("Error: 'semantic_vector' column not found.")
        sys.exit(1)

    # 3. Encode 'label' column
    print("Encoding labels...")
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    print(f"Labels mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    label_classes = le.classes_

    # 4. Handle Lexical Features
    lexical_cols = ['word_count', 'unique_word_ratio', 'profanity_score']
    for col in lexical_cols:
        if col not in df.columns:
            print(f"Warning: '{col}' column not found. Setting to 0.0.")
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 5. Handle Syntactic Features
    syntactic_cols = [col for col in df.columns if col.startswith('pos_')]
    if syntactic_cols:
        print(f"Found {len(syntactic_cols)} syntactic columns.")
        for col in syntactic_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    else:
        print("Warning: No syntactic (POS) columns found.")

    # 6. Handle TF-IDF features
    tfidf_sparse_matrix = None
    if not tfidf_sparse_path.exists():
        print(f"Warning: Sparse TF-IDF matrix file not found at {tfidf_sparse_path}.")
    else:
        try:
            tfidf_sparse_matrix = load_npz(tfidf_sparse_path)
            print(f"Loaded sparse TF-IDF matrix with shape: {tfidf_sparse_matrix.shape}")
            if tfidf_sparse_matrix.shape[0] != len(df):
                print(f"Error: Row count mismatch between data ({len(df)}) and sparse TF-IDF ({tfidf_sparse_matrix.shape[0]}).")
                # Attempt to reconcile if possible, otherwise exit
                # Example: Filter df to match tfidf_sparse_matrix indices if df is larger
                # Or reload data ensuring consistency
                sys.exit(1) # Exit for now, requires careful handling
        except Exception as e:
            print(f"Error loading sparse TF-IDF matrix: {e}")

    tfidf_cols_dense = [col for col in df.columns if col.startswith('tfidf_')]
    if tfidf_cols_dense:
        print(f"Found {len(tfidf_cols_dense)} dense TF-IDF columns.")
        for col in tfidf_cols_dense:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    else:
        print("Warning: No dense TF-IDF columns found.")

    # Remove potentially problematic list-based tfidf columns
    for col_name in ['tfidf', 'tfidf_feature']:
        if col_name in df.columns:
            df = df.drop(columns=[col_name])

    df.dropna(subset=['clean_text', 'label_encoded'], inplace=True)
    print(f"Data shape after final processing: {df.shape}")

    return df, tfidf_sparse_matrix, label_classes

# --- Feature Preparation Helper (Keep as before) ---

def prepare_features(df, feature_cols, scaler=None, fit_scaler=False, is_sparse=False):
    """Extracts, handles NaNs, and optionally scales features."""
    if is_sparse:
        if df is None:
            raise ValueError("Sparse matrix not provided or loaded correctly.")
        return df, None

    X = df[feature_cols].copy()
    if X.isnull().values.any():
        print(f"  Warning: NaN values found in features {feature_cols}. Imputing with 0.")
        X.fillna(0, inplace=True)
    X_np = X.values.astype(np.float32)

    current_scaler = scaler if scaler is not None else StandardScaler()

    if fit_scaler:
        X_scaled = current_scaler.fit_transform(X_np)
        # print(f"    Scaler fitted and data scaled. Feature shape: {X_scaled.shape}") # Less verbose
        return X_scaled, current_scaler # Return the fitted scaler
    else:
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_np)
                # print(f"    Data scaled using existing scaler. Feature shape: {X_scaled.shape}") # Less verbose
            except Exception as e:
                print(f"    Warning: Scaling failed: {e}. Using unscaled data.")
                X_scaled = X_np
        else:
            # print("    No scaler provided, using unscaled data.") # Less verbose
            X_scaled = X_np
        return X_scaled, scaler # Return the original scaler (or None)


# --- Evaluation Function (Keep as before) ---
def evaluate_model(model, X_val, y_val, is_keras=False, batch_size=BATCH_SIZE):
    """Evaluates a model and returns metrics."""
    try:
        y_pred_proba = None
        y_pred = None
        if is_keras:
            y_pred_proba = model.predict(X_val, batch_size=batch_size).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_val)
            y_pred = (y_scores > 0).astype(int)
            if isinstance(model, CalibratedClassifierCV):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                print("Warning: Using decision_function scores instead of probabilities.")
                # y_pred_proba = y_scores # Or sigmoid
        else:
            y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)
        return accuracy, precision, recall, f1, y_pred_proba, y_pred
    except Exception as e:
        print(f"      ERROR during evaluation: {e}")
        return np.nan, np.nan, np.nan, np.nan, None, None

# --- Cross-Validation Function (Keep as before) ---

def cross_validate_fusion_strategies(df_train, tfidf_sparse_matrix_train, label_classes):
    """Performs cross-validation on the training set to find the best strategy."""
    print("\n===== Starting Cross-Validation on Training Set =====")
    # --- Define Feature Sets ---
    lexical_features = [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df_train.columns]
    syntactic_features = [col for col in df_train.columns if col.startswith('pos_')]
    sentiment_features = [col for col in df_train.columns if col.startswith('sentiment_')]
    tfidf_dense_features = [col for col in df_train.columns if col.startswith('tfidf_')]

    # --- Target Variable ---
    y_train_full = df_train['label_encoded'].values

    # --- Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    # --- Keras Early Stopping ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)

    # --- Cross-Validation Loop ---
    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(df_train, y_train_full)):
        print(f"\n{'='*15} CV FOLD {fold+1}/{N_SPLITS} {'='*15}")

        df_train_fold, df_val_fold = df_train.iloc[train_fold_idx], df_train.iloc[val_fold_idx]
        y_train_fold, y_val_fold = y_train_full[train_fold_idx], y_train_full[val_fold_idx]

        # --- Prepare Base Feature Sets for this Fold ---
        # TF-IDF (Sparse)
        X_train_fold_tfidf_sparse, X_val_fold_tfidf_sparse = None, None
        if tfidf_sparse_matrix_train is not None:
            X_train_fold_tfidf_sparse = tfidf_sparse_matrix_train[train_fold_idx]
            X_val_fold_tfidf_sparse = tfidf_sparse_matrix_train[val_fold_idx]
            # print(f"  Prepared Sparse TF-IDF: Train {X_train_fold_tfidf_sparse.shape}, Val {X_val_fold_tfidf_sparse.shape}") # Less verbose

        # TF-IDF (Dense)
        X_train_fold_tfidf_dense, X_val_fold_tfidf_dense, scaler_tfidf_dense = None, None, None
        if tfidf_dense_features:
            X_train_fold_tfidf_dense, scaler_tfidf_dense = prepare_features(df_train_fold, tfidf_dense_features, fit_scaler=True)
            X_val_fold_tfidf_dense, _ = prepare_features(df_val_fold, tfidf_dense_features, scaler=scaler_tfidf_dense)
            # print(f"  Prepared Dense TF-IDF: Train {X_train_fold_tfidf_dense.shape}, Val {X_val_fold_tfidf_dense.shape}") # Less verbose

        # Lexical
        X_train_fold_lex, X_val_fold_lex, scaler_lex = None, None, None
        if lexical_features:
            X_train_fold_lex, scaler_lex = prepare_features(df_train_fold, lexical_features, fit_scaler=True)
            X_val_fold_lex, _ = prepare_features(df_val_fold, lexical_features, scaler=scaler_lex)
            # print(f"  Prepared Lexical: Train {X_train_fold_lex.shape}, Val {X_val_fold_lex.shape}") # Less verbose

        # Syntactic
        X_train_fold_syn, X_val_fold_syn, scaler_syn = None, None, None
        if syntactic_features:
            X_train_fold_syn, scaler_syn = prepare_features(df_train_fold, syntactic_features, fit_scaler=True)
            X_val_fold_syn, _ = prepare_features(df_val_fold, syntactic_features, scaler=scaler_syn)
            # print(f"  Prepared Syntactic: Train {X_train_fold_syn.shape}, Val {X_val_fold_syn.shape}") # Less verbose

        # Semantic
        X_train_fold_sem, X_val_fold_sem, scaler_sem = None, None, None
        if 'semantic_vector' in df_train.columns:
            X_train_fold_sem_raw = np.stack(df_train_fold['semantic_vector'].values)
            X_val_fold_sem_raw = np.stack(df_val_fold['semantic_vector'].values)
            scaler_sem = StandardScaler()
            X_train_fold_sem = scaler_sem.fit_transform(X_train_fold_sem_raw)
            X_val_fold_sem = scaler_sem.transform(X_val_fold_sem_raw)
            # print(f"  Prepared Semantic: Train {X_train_fold_sem.shape}, Val {X_val_fold_sem.shape}") # Less verbose

        # --- Evaluate Strategies for this Fold ---
        strategies_to_run = [
            'Early (TFIDF_Dense+Lex+Syn -> SVM)',
            'Intermediate (Sem+Syn -> BiLSTM)',
            'Late Fusion (Avg Prob)',
            'Baseline (TF-IDF Sparse -> SVM)',
            'Baseline (Semantic -> Dense NN)'
        ]

        for strategy_name in strategies_to_run:
            # print(f"\n  --- Evaluating Strategy: {strategy_name} ---") # Less verbose
            acc, pre, rec, f1 = np.nan, np.nan, np.nan, np.nan # Default to NaN

            try:
                if strategy_name == 'Early (TFIDF_Dense+Lex+Syn -> SVM)':
                    if X_train_fold_tfidf_dense is not None and X_train_fold_lex is not None and X_train_fold_syn is not None:
                        X_train_early = np.hstack((X_train_fold_tfidf_dense, X_train_fold_lex, X_train_fold_syn))
                        X_val_early = np.hstack((X_val_fold_tfidf_dense, X_val_fold_lex, X_val_fold_syn))
                        model_early = build_svm(kernel='rbf', probability=True, random_state=RANDOM_STATE + fold)
                        model_early.fit(X_train_early, y_train_fold)
                        acc, pre, rec, f1, _, _ = evaluate_model(model_early, X_val_early, y_val_fold)
                        del model_early, X_train_early, X_val_early
                    # else: print("    Skipping: Missing required dense features.") # Less verbose

                elif strategy_name == 'Intermediate (Sem+Syn -> BiLSTM)':
                    if X_train_fold_sem is not None and X_train_fold_syn is not None:
                        current_semantic_dim = X_train_fold_sem.shape[1]
                        current_syntactic_dim = X_train_fold_syn.shape[1]
                        model_intermediate = build_bilstm_intermediate(current_semantic_dim, current_syntactic_dim)
                        history = model_intermediate.fit(
                            [X_train_fold_sem, X_train_fold_syn], y_train_fold,
                            validation_data=([X_val_fold_sem, X_val_fold_syn], y_val_fold),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0
                        )
                        acc, pre, rec, f1, _, _ = evaluate_model(model_intermediate, [X_val_fold_sem, X_val_fold_syn], y_val_fold, is_keras=True)
                        tf.keras.backend.clear_session()
                        del model_intermediate, history
                    # else: print("    Skipping: Missing Semantic or Syntactic features.") # Less verbose

                elif strategy_name == 'Late Fusion (Avg Prob)':
                    if X_train_fold_tfidf_sparse is not None and X_train_fold_sem is not None:
                        # Model 1: SVM
                        model_late_svm = build_svm(kernel='linear', probability=True, random_state=RANDOM_STATE + fold)
                        model_late_svm.fit(X_train_fold_tfidf_sparse, y_train_fold)
                        _, _, _, _, svm_probs, _ = evaluate_model(model_late_svm, X_val_fold_tfidf_sparse, y_val_fold)
                        # Model 2: NN
                        tf.random.set_seed(RANDOM_STATE + fold)
                        model_late_nn = build_dense_for_semantic(input_dim=X_train_fold_sem.shape[1])
                        history_nn = model_late_nn.fit(X_train_fold_sem, y_train_fold,
                                                       epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                       validation_data=(X_val_fold_sem, y_val_fold),
                                                       callbacks=[early_stopping], verbose=0)
                        _, _, _, _, nn_probs, _ = evaluate_model(model_late_nn, X_val_fold_sem, y_val_fold, is_keras=True)

                        if svm_probs is not None and nn_probs is not None:
                            avg_probs = (svm_probs + nn_probs) / 2
                            y_pred_late = (avg_probs > 0.5).astype(int)
                            acc = accuracy_score(y_val_fold, y_pred_late)
                            pre, rec, f1, _ = precision_recall_fscore_support(y_val_fold, y_pred_late, average='binary', zero_division=0)
                        # else: print("    Skipping evaluation due to missing probabilities.") # Less verbose
                        tf.keras.backend.clear_session()
                        del model_late_svm, model_late_nn, svm_probs, nn_probs, history_nn
                    # else: print("    Skipping: Missing Sparse TF-IDF or Semantic features.") # Less verbose

                elif strategy_name == 'Baseline (TF-IDF Sparse -> SVM)':
                    if X_train_fold_tfidf_sparse is not None:
                        model_base_svm = build_svm(kernel='linear', probability=False, random_state=RANDOM_STATE + fold)
                        model_base_svm.fit(X_train_fold_tfidf_sparse, y_train_fold)
                        acc, pre, rec, f1, _, _ = evaluate_model(model_base_svm, X_val_fold_tfidf_sparse, y_val_fold)
                        del model_base_svm
                    # else: print("    Skipping: Missing Sparse TF-IDF features.") # Less verbose

                elif strategy_name == 'Baseline (Semantic -> Dense NN)':
                    if X_train_fold_sem is not None:
                        tf.random.set_seed(RANDOM_STATE + fold)
                        model_base_nn = build_dense_for_semantic(input_dim=X_train_fold_sem.shape[1])
                        history_base_nn = model_base_nn.fit(X_train_fold_sem, y_train_fold,
                                                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                            validation_data=(X_val_fold_sem, y_val_fold),
                                                            callbacks=[early_stopping], verbose=0)
                        acc, pre, rec, f1, _, _ = evaluate_model(model_base_nn, X_val_fold_sem, y_val_fold, is_keras=True)
                        tf.keras.backend.clear_session()
                        del model_base_nn, history_base_nn
                    # else: print("    Skipping: Missing Semantic features.") # Less verbose

                # Record results for the strategy in this fold
                # print(f"    {strategy_name} Metrics: Acc={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}") # Less verbose
                all_results.append({'fold': fold+1, 'strategy': strategy_name, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1_score': f1})

            except Exception as e:
                print(f"    ERROR during strategy '{strategy_name}': {e}")
                all_results.append({'fold': fold+1, 'strategy': strategy_name, 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
                if 'model_intermediate' in locals() or 'model_late_nn' in locals() or 'model_base_nn' in locals():
                    tf.keras.backend.clear_session() # Clear session if Keras model was involved in error
            finally:
                gc.collect() # Force garbage collection after each strategy

        # Clean up fold-specific data
        del df_train_fold, df_val_fold, y_train_fold, y_val_fold
        del X_train_fold_tfidf_sparse, X_val_fold_tfidf_sparse
        if tfidf_dense_features: del X_train_fold_tfidf_dense, X_val_fold_tfidf_dense, scaler_tfidf_dense
        if lexical_features: del X_train_fold_lex, X_val_fold_lex, scaler_lex
        if syntactic_features: del X_train_fold_syn, X_val_fold_syn, scaler_syn
        if 'semantic_vector' in df_train.columns: del X_train_fold_sem, X_val_fold_sem, scaler_sem, X_train_fold_sem_raw, X_val_fold_sem_raw
        gc.collect()

    print("\n===== Cross-Validation Finished =====")
    results_df = pd.DataFrame(all_results)
    return results_df

# --- Function to Save Final Model (Keep as before) ---
def save_final_model(model, strategy_name, output_dir, model_name="final_model"):
    """Saves the final trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize strategy name for filename
    safe_strategy_name = "".join(c if c.isalnum() else "_" for c in strategy_name)
    filename_base = f"{safe_strategy_name}_{model_name}"

    if isinstance(model, (list, tuple)) and 'Late Fusion' in strategy_name:
        # Save both models for late fusion
        model_svm, model_nn = model
        path_svm = output_dir / f"{filename_base}_svm.pkl"
        path_nn = output_dir / f"{filename_base}_nn.h5"
        try:
            joblib.dump(model_svm, path_svm)
            model_nn.save(path_nn)
            print(f"Late fusion models saved: {path_svm}, {path_nn}")
            return str(path_svm), str(path_nn) # Return paths
        except Exception as e:
            print(f"Error saving late fusion models: {e}")
            return None, None
    elif hasattr(model, 'save'): # Keras model
        path = output_dir / f"{filename_base}.h5"
        try:
            model.save(path)
            print(f"Final Keras model saved: {path}")
            return str(path)
        except Exception as e:
            print(f"Error saving Keras model {path}: {e}")
            return None
    else: # Scikit-learn model
        path = output_dir / f"{filename_base}.pkl"
        try:
            joblib.dump(model, path)
            print(f"Final scikit-learn model saved: {path}")
            return str(path)
        except Exception as e:
            print(f"Error saving scikit-learn model {path}: {e}")
            return None

# --- Main Execution (Modified) ---
if __name__ == "__main__":
    # Define paths
    data_dir = Path('data/processed')
    input_file = data_dir / 'phase2_output.csv'
    tfidf_sparse_file = data_dir / 'tfidf_matrix.npz'
    output_dir = Path('results/phase4')
    cv_output_file = output_dir / 'cv_fusion_results.csv'
    cv_summary_file = output_dir / 'cv_fusion_summary.csv'
    test_output_file = output_dir / 'test_set_fusion_results.csv' # New file for test results
    final_model_dir = Path('models/phase4') # Directory to save final models

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    df, tfidf_sparse_matrix, label_classes = load_and_preprocess_data(input_file, tfidf_sparse_file)
    if df.empty:
        print("Error: DataFrame is empty after preprocessing. Cannot proceed.")
        sys.exit(1)

    # 2. Split data into Training and Test sets
    print(f"\nSplitting data into Train ({1-TEST_SET_SIZE:.0%}) and Test ({TEST_SET_SIZE:.0%})...")
    train_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label_encoded'].values
    )
    df_train = df.iloc[train_indices].copy().reset_index(drop=True) # Reset index for easier handling
    df_test = df.iloc[test_indices].copy().reset_index(drop=True)
    y_train_full = df_train['label_encoded'].values
    y_test = df_test['label_encoded'].values

    tfidf_sparse_matrix_train = None
    tfidf_sparse_matrix_test = None
    if tfidf_sparse_matrix is not None:
        tfidf_sparse_matrix_train = tfidf_sparse_matrix[train_indices]
        tfidf_sparse_matrix_test = tfidf_sparse_matrix[test_indices]
        print(f"Split sparse TF-IDF: Train {tfidf_sparse_matrix_train.shape}, Test {tfidf_sparse_matrix_test.shape}")

    print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")

    # 3. Perform Cross-Validation on the Training Set
    cv_results_df = cross_validate_fusion_strategies(df_train, tfidf_sparse_matrix_train, label_classes)

    # 4. Save and Analyze CV Results
    if not cv_results_df.empty:
        print(f"\nSaving detailed CV results to: {cv_output_file}")
        cv_results_df.to_csv(cv_output_file, index=False)

        print("\n--- Average CV Performance Across Folds (Grouped by Strategy) ---")
        avg_cv_results = cv_results_df.groupby('strategy')[['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean', 'std'])
        avg_cv_results.columns = ['_'.join(col).strip() for col in avg_cv_results.columns.values]
        avg_cv_results = avg_cv_results.rename(columns={
            'accuracy_mean': 'Avg Accuracy', 'accuracy_std': 'Std Dev Acc',
            'precision_mean': 'Avg Precision', 'precision_std': 'Std Dev Prec',
            'recall_mean': 'Avg Recall', 'recall_std': 'Std Dev Rec',
            'f1_score_mean': 'Avg F1-Score', 'f1_score_std': 'Std Dev F1'
        })
        avg_cv_results = avg_cv_results.sort_values(by='Avg F1-Score', ascending=False)
        print(avg_cv_results)
        print(f"\nSaving CV summary results to: {cv_summary_file}")
        avg_cv_results.to_csv(cv_summary_file)

        # Get list of strategies evaluated during CV
        evaluated_strategies = cv_results_df['strategy'].unique()

    else:
        print("\nNo CV results were generated. Cannot proceed with final model training.")
        sys.exit(1)


    # 5. Train Final Model for EACH Strategy on Entire Training Set and Evaluate on Test Set
    print(f"\n===== Training Final Models on Full Training Set & Evaluating on Test Set =====")
    all_test_results = []
    saved_model_paths = {} # Store paths of saved models

    # Define feature sets based on full training data
    lexical_features = [col for col in df_train.columns if col in ['word_count', 'unique_word_ratio', 'profanity_score']]
    syntactic_features = [col for col in df_train.columns if col.startswith('pos_')]
    tfidf_dense_features = [col for col in df_train.columns if col.startswith('tfidf_')]

    # Prepare scalers fitted on the full training data ONCE
    scalers = {}
    if tfidf_dense_features:
        _, scalers['tfidf_dense'] = prepare_features(df_train, tfidf_dense_features, fit_scaler=True)
    if lexical_features:
        _, scalers['lex'] = prepare_features(df_train, lexical_features, fit_scaler=True)
    if syntactic_features:
        _, scalers['syn'] = prepare_features(df_train, syntactic_features, fit_scaler=True)
    if 'semantic_vector' in df_train.columns:
        X_train_sem_raw_full = np.stack(df_train['semantic_vector'].values)
        scalers['sem'] = StandardScaler().fit(X_train_sem_raw_full)
        # Save scalers immediately
        for name, scaler in scalers.items():
            joblib.dump(scaler, final_model_dir / f'scaler_{name}.pkl')
        print("Saved scalers fitted on full training data.")


    for strategy_name in evaluated_strategies:
        print(f"\n--- Processing Strategy for Final Training & Test: {strategy_name} ---")
        final_model = None
        test_features = None
        is_keras_final = False
        acc_test, pre_test, rec_test, f1_test = np.nan, np.nan, np.nan, np.nan

        try:
            # Prepare features for this strategy using full train/test sets and fitted scalers
            if strategy_name == 'Early (TFIDF_Dense+Lex+Syn -> SVM)':
                if tfidf_dense_features and lexical_features and syntactic_features:
                    X_train_tfidf_dense, _ = prepare_features(df_train, tfidf_dense_features, scaler=scalers.get('tfidf_dense'))
                    X_test_tfidf_dense, _ = prepare_features(df_test, tfidf_dense_features, scaler=scalers.get('tfidf_dense'))
                    X_train_lex, _ = prepare_features(df_train, lexical_features, scaler=scalers.get('lex'))
                    X_test_lex, _ = prepare_features(df_test, lexical_features, scaler=scalers.get('lex'))
                    X_train_syn, _ = prepare_features(df_train, syntactic_features, scaler=scalers.get('syn'))
                    X_test_syn, _ = prepare_features(df_test, syntactic_features, scaler=scalers.get('syn'))

                    X_train_final = np.hstack((X_train_tfidf_dense, X_train_lex, X_train_syn))
                    test_features = np.hstack((X_test_tfidf_dense, X_test_lex, X_test_syn))

                    final_model = build_svm(kernel='rbf', probability=True, random_state=RANDOM_STATE)
                    print("  Training final Early Fusion SVM...")
                    final_model.fit(X_train_final, y_train_full)
                else: print("  Cannot train final Early Fusion model: Missing required features.")

            elif strategy_name == 'Intermediate (Sem+Syn -> BiLSTM)':
                if 'semantic_vector' in df_train.columns and syntactic_features:
                    X_train_sem_raw = np.stack(df_train['semantic_vector'].values)
                    X_test_sem_raw = np.stack(df_test['semantic_vector'].values)
                    X_train_sem = scalers['sem'].transform(X_train_sem_raw)
                    X_test_sem = scalers['sem'].transform(X_test_sem_raw)

                    X_train_syn, _ = prepare_features(df_train, syntactic_features, scaler=scalers.get('syn'))
                    X_test_syn, _ = prepare_features(df_test, syntactic_features, scaler=scalers.get('syn'))

                    test_features = [X_test_sem, X_test_syn]
                    is_keras_final = True

                    final_model = build_bilstm_intermediate(X_train_sem.shape[1], X_train_syn.shape[1])
                    print("  Training final Intermediate Fusion Bi-LSTM...")
                    final_model.fit([X_train_sem, X_train_syn], y_train_full,
                                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0) # Reduced verbosity
                else: print("  Cannot train final Intermediate Fusion model: Missing required features.")

            elif strategy_name == 'Late Fusion (Avg Prob)':
                if tfidf_sparse_matrix_train is not None and 'semantic_vector' in df_train.columns:
                    # Train SVM model
                    model_svm = build_svm(kernel='linear', probability=True, random_state=RANDOM_STATE)
                    print("  Training final Late Fusion SVM...")
                    model_svm.fit(tfidf_sparse_matrix_train, y_train_full)

                    # Train NN model
                    X_train_sem_raw = np.stack(df_train['semantic_vector'].values)
                    X_test_sem_raw = np.stack(df_test['semantic_vector'].values)
                    X_train_sem = scalers['sem'].transform(X_train_sem_raw)
                    X_test_sem = scalers['sem'].transform(X_test_sem_raw)

                    tf.random.set_seed(RANDOM_STATE)
                    model_nn = build_dense_for_semantic(input_dim=X_train_sem.shape[1])
                    print("  Training final Late Fusion Dense NN...")
                    # Consider adding early stopping based on a small validation split from training data if needed
                    model_nn.fit(X_train_sem, y_train_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

                    final_model = (model_svm, model_nn) # Store tuple of models
                    test_features = (tfidf_sparse_matrix_test, X_test_sem) # Tuple of test features
                else: print("  Cannot train final Late Fusion models: Missing required features.")

            elif strategy_name == 'Baseline (TF-IDF Sparse -> SVM)':
                if tfidf_sparse_matrix_train is not None:
                    final_model = build_svm(kernel='linear', probability=False, random_state=RANDOM_STATE)
                    print("  Training final Baseline SVM...")
                    final_model.fit(tfidf_sparse_matrix_train, y_train_full)
                    test_features = tfidf_sparse_matrix_test
                else: print("  Cannot train final Baseline SVM: Missing TF-IDF features.")

            elif strategy_name == 'Baseline (Semantic -> Dense NN)':
                if 'semantic_vector' in df_train.columns:
                    X_train_sem_raw = np.stack(df_train['semantic_vector'].values)
                    X_test_sem_raw = np.stack(df_test['semantic_vector'].values)
                    X_train_sem = scalers['sem'].transform(X_train_sem_raw)
                    X_test_sem = scalers['sem'].transform(X_test_sem_raw)

                    test_features = X_test_sem
                    is_keras_final = True

                    tf.random.set_seed(RANDOM_STATE)
                    final_model = build_dense_for_semantic(input_dim=X_train_sem.shape[1])
                    print("  Training final Baseline Dense NN...")
                    final_model.fit(X_train_sem, y_train_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                else: print("  Cannot train final Baseline NN: Missing Semantic features.")

            # Evaluate the trained final model on the test set
            if final_model is not None and test_features is not None:
                print(f"  Evaluating final model on Test Set...")

                if strategy_name == 'Late Fusion (Avg Prob)':
                    model_svm, model_nn = final_model
                    X_test_tfidf, X_test_sem_scaled = test_features
                    _, _, _, _, svm_probs_test, _ = evaluate_model(model_svm, X_test_tfidf, y_test)
                    _, _, _, _, nn_probs_test, _ = evaluate_model(model_nn, X_test_sem_scaled, y_test, is_keras=True)

                    if svm_probs_test is not None and nn_probs_test is not None:
                        avg_probs_test = (svm_probs_test + nn_probs_test) / 2
                        y_pred_test = (avg_probs_test > 0.5).astype(int)
                        acc_test = accuracy_score(y_test, y_pred_test)
                        pre_test, rec_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary', zero_division=0)
                    else:
                        print("  Evaluation failed due to missing probabilities.")
                else:
                    acc_test, pre_test, rec_test, f1_test, _, y_pred_test = evaluate_model(
                        final_model, test_features, y_test, is_keras=is_keras_final
                    )

                print(f"  Test Set Metrics: Acc={acc_test:.4f}, P={pre_test:.4f}, R={rec_test:.4f}, F1={f1_test:.4f}")
                # Optional: Print classification report for test set
                # if y_pred_test is not None:
                #     print("\n  Test Classification Report:")
                #     print(classification_report(y_test, y_pred_test, target_names=[str(c) for c in label_classes], zero_division=0))

                # Save the final model for this strategy
                print(f"  Saving final model for {strategy_name}...")
                model_paths = save_final_model(final_model, strategy_name, final_model_dir)
                if model_paths:
                    saved_model_paths[strategy_name] = model_paths

            else:
                print("  Could not train or evaluate the final model for this strategy.")

        except Exception as e:
            print(f"  ERROR during final training/evaluation for strategy '{strategy_name}': {e}")
            if 'final_model' in locals() and hasattr(final_model, 'save'): # Check if Keras model exists
                tf.keras.backend.clear_session() # Clear session if Keras model was involved in error

        finally:
            # Store test results (even if NaN)
            all_test_results.append({
                'strategy': strategy_name,
                'test_accuracy': acc_test,
                'test_precision': pre_test,
                'test_recall': rec_test,
                'test_f1_score': f1_test
            })
            # Clean up memory
            if 'final_model' in locals(): del final_model
            if 'test_features' in locals(): del test_features
            gc.collect()


    # 6. Save Test Set Results
    if all_test_results:
        test_results_df = pd.DataFrame(all_test_results)
        print(f"\nSaving test set results for all strategies to: {test_output_file}")
        test_results_df.to_csv(test_output_file, index=False)

        # 7. Identify and Report Best Model based on Test Set F1
        test_results_df = test_results_df.sort_values(by='test_f1_score', ascending=False).reset_index(drop=True)
        print("\n--- Final Test Set Performance Summary (Sorted by F1-Score) ---")
        print(test_results_df)

        if not test_results_df.empty and not pd.isna(test_results_df.iloc[0]['test_f1_score']):
            best_test_strategy = test_results_df.iloc[0]['strategy']
            best_test_f1 = test_results_df.iloc[0]['test_f1_score']
            print(f"\nOverall Best Strategy based on Test F1-Score: {best_test_strategy} (F1 = {best_test_f1:.4f})")
            print(f"Model(s) saved in: {final_model_dir}")
            if best_test_strategy in saved_model_paths:
                print(f"Specific model file(s): {saved_model_paths[best_test_strategy]}")
        else:
            print("\nCould not determine the best strategy based on test results (results might be empty or NaN).")

    else:
        print("\nNo test results were generated.")


    print("\nPhase 4 Processing complete.")