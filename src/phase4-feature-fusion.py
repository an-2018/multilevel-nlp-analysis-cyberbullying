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
from sklearn.linear_model import LogisticRegression # Added for potential calibration
from sklearn.calibration import CalibratedClassifierCV

# TensorFlow/Keras imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow verbosity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalMaxPooling1D, Dropout, Bidirectional, concatenate, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- Constants ---
N_SPLITS = 5
RANDOM_STATE = 42
SEMANTIC_DIM = 384 # Dimension of 'all-MiniLM-L6-v2' embeddings
EPOCHS = 50 # Max epochs for Keras models
BATCH_SIZE = 32
PATIENCE = 5 # Early stopping patience

# --- Model Building Functions (Adapted/Added from Phase 3) ---

def build_svm(kernel='linear', probability=False, class_weight='balanced', random_state=None):
    """Builds a Support Vector Classifier."""
    if kernel == 'linear':
        # LinearSVC is often faster for high-dimensional sparse data like TF-IDF
        # Use CalibratedClassifierCV to get probabilities if needed for late fusion
        base_svm = LinearSVC(
            class_weight=class_weight,
            dual='auto', # Let sklearn choose based on n_samples/n_features
            random_state=random_state,
            max_iter=5000,
            C=1.0,
            tol=1e-3
        )
        if probability:
            # Use CalibratedClassifierCV for probability estimates with LinearSVC
            # 'isotonic' often works well but requires more data; 'sigmoid' is faster
            return CalibratedClassifierCV(base_svm, method='sigmoid', cv=3) # Use 3-fold internal CV for calibration
        else:
            return base_svm
    else:
        # For non-linear kernels (like RBF), use SVC directly
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
    # Input layers
    semantic_input = Input(shape=(semantic_dim,), name='semantic_input')
    syntactic_input = Input(shape=(syntactic_dim,), name='syntactic_input')

    # Process semantic input (assuming it's a flat vector, reshape for LSTM)
    # If semantic_dim is large, consider a Dense layer first for dimensionality reduction
    # Reshape semantic input to be suitable for LSTM (e.g., treat as a sequence of 1 step)
    semantic_reshaped = tf.keras.layers.Reshape((1, semantic_dim))(semantic_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(semantic_reshaped) # Or adjust units

    # Concatenate LSTM output with syntactic features
    combined = concatenate([lstm_out, syntactic_input])

    # Dense layers for classification
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

# --- Data Loading and Preprocessing (Adapted from Phase 3) ---

def safe_literal_eval(x):
    """Safely evaluates a string literal, returning an empty list on error."""
    try:
        # Ensure it's a string before attempting eval
        if isinstance(x, str):
            val = ast.literal_eval(x)
            # Ensure it's a list or tuple before converting to numpy array
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.float32) # Specify dtype for consistency
        return np.array([], dtype=np.float32) # Return empty array of correct type
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32) # Return empty array on error

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

        # Check for empty arrays resulting from parsing errors or empty text
        empty_vectors = df['semantic_vector'].apply(lambda x: x.size == 0)
        if empty_vectors.any():
            print(f"Warning: Found {empty_vectors.sum()} rows with invalid/empty semantic vectors. These rows might cause issues or be dropped.")
            # Option 1: Fill with zeros (might skew results)
            # df.loc[empty_vectors, 'semantic_vector'] = df.loc[empty_vectors, 'semantic_vector'].apply(lambda x: np.zeros(SEMANTIC_DIM))
            # Option 2: Drop rows (safer if only a few)
            print(f"Dropping {empty_vectors.sum()} rows with invalid semantic vectors.")
            df = df[~empty_vectors].copy() # Use .copy() to avoid SettingWithCopyWarning
            df.reset_index(drop=True, inplace=True)
            if df.empty:
                print("Error: DataFrame is empty after removing rows with invalid semantic vectors.")
                sys.exit(1)

        # Verify the dimension of the first valid vector
        if not df.empty:
            first_valid_vector_dim = df['semantic_vector'].iloc[0].shape[0]
            if first_valid_vector_dim != SEMANTIC_DIM:
                print(f"Warning: Expected semantic vector dimension {SEMANTIC_DIM}, but found {first_valid_vector_dim}. Using actual dimension: {first_valid_vector_dim}")
                # Update SEMANTIC_DIM globally if you want to use the detected dimension
                # global SEMANTIC_DIM
                # SEMANTIC_DIM = first_valid_vector_dim
            # Check consistency across all vectors
            inconsistent_dims = df['semantic_vector'].apply(lambda x: x.shape[0] != first_valid_vector_dim).sum()
            if inconsistent_dims > 0:
                print(f"Error: Found {inconsistent_dims} semantic vectors with inconsistent dimensions. Please check data generation.")
                sys.exit(1)
    else:
        print("Error: 'semantic_vector' column not found. Required for semantic and fusion models.")
        sys.exit(1)


    # 3. Encode 'label' column
    print("Encoding labels...")
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    print(f"Labels mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    label_classes = le.classes_

    # 4. Handle potentially missing 'profanity_score' and other lexical features
    lexical_cols = ['word_count', 'unique_word_ratio', 'profanity_score'] # Add others if needed
    for col in lexical_cols:
        if col not in df.columns:
            print(f"Warning: '{col}' column not found. Setting to 0.0.")
            df[col] = 0.0
        else:
            # Ensure numeric and handle NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"Warning: Column {col} contained non-numeric values converted to NaN. Filling with 0.")
                df[col].fillna(0, inplace=True)

    # 5. Handle Syntactic Features (POS tags)
    syntactic_cols = [col for col in df.columns if col.startswith('pos_')]
    if not syntactic_cols:
        print("Warning: No syntactic (POS) columns found. Intermediate fusion might be affected.")
    else:
        print(f"Found {len(syntactic_cols)} syntactic columns.")
        for col in syntactic_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        print(f"Warning: Column {col} contained non-numeric values converted to NaN. Filling with 0.")
                        df[col].fillna(0, inplace=True)
                except Exception as e:
                    print(f"Error converting column {col} to numeric: {e}. Check data quality.")
                    # Decide how to handle: drop column, exit, etc.
                    # df = df.drop(columns=[col])
                    # sys.exit(1)

    # 6. Handle TF-IDF features (check for dense columns, assume sparse is separate)
    tfidf_sparse_matrix = None
    if not tfidf_sparse_path.exists():
        print(f"Warning: Sparse TF-IDF matrix file not found at {tfidf_sparse_path}. 'tfidf_sparse' dependent models will be skipped.")
    else:
        try:
            tfidf_sparse_matrix = load_npz(tfidf_sparse_path)
            print(f"Loaded sparse TF-IDF matrix with shape: {tfidf_sparse_matrix.shape}")
            if tfidf_sparse_matrix.shape[0] != len(df):
                print(f"Error: Row count mismatch between loaded data ({len(df)}) and sparse TF-IDF matrix ({tfidf_sparse_matrix.shape[0]}). Ensure they correspond.")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading sparse TF-IDF matrix: {e}")
            tfidf_sparse_matrix = None # Ensure it's None if loading fails

    tfidf_cols_dense = [col for col in df.columns if col.startswith('tfidf_')]
    if not tfidf_cols_dense:
        print("Warning: No dense TF-IDF columns (starting with 'tfidf_') found. 'tfidf_dense' group will be skipped if used.")
    else:
        print(f"Found {len(tfidf_cols_dense)} dense TF-IDF columns ('tfidf_*'). Converting to numeric.")
        for col in tfidf_cols_dense:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        print(f"Warning: Column {col} contained non-numeric values converted to NaN. Filling with 0.")
                        df[col].fillna(0, inplace=True)
                except Exception as e:
                    print(f"Error converting column {col} to numeric: {e}. Check data quality.")
                    # sys.exit(1) # Or handle differently

    # Remove potentially problematic list-based tfidf columns from previous runs if they exist
    for col_name in ['tfidf', 'tfidf_feature']: # Adjust if your column names differ
        if col_name in df.columns:
            print(f"Removing potentially problematic column '{col_name}'.")
            df = df.drop(columns=[col_name])

    # Final check for NaNs in critical columns
    df.dropna(subset=['clean_text', 'label_encoded'], inplace=True)
    print(f"Data shape after final processing and NaN drop: {df.shape}")

    return df, tfidf_sparse_matrix, label_classes

# --- Feature Preparation Helper ---

def prepare_features(df, feature_cols, scaler=None, fit_scaler=False, is_sparse=False):
    """Extracts, handles NaNs, and optionally scales features."""
    if is_sparse:
        # For sparse data, we assume it's already prepared (like loaded from .npz)
        # We just return it directly. The input 'df' in this case should be the sparse matrix itself.
        if df is None:
            raise ValueError("Sparse matrix not provided or loaded correctly.")
        return df, None # Return None for scaler as scaling is typically not applied to sparse TF-IDF

    # For dense features
    X = df[feature_cols].copy()

    # Handle potential NaNs that might have slipped through or resulted from merges/ops
    if X.isnull().values.any():
        print(f"  Warning: NaN values found in selected features {feature_cols}. Imputing with 0.")
        X.fillna(0, inplace=True)

    # Convert to numpy array
    X_np = X.values.astype(np.float32)

    # Scale data
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X_np)
        print(f"    Scaler fitted and data scaled. Feature shape: {X_scaled.shape}")
    else:
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_np)
                print(f"    Data scaled using existing scaler. Feature shape: {X_scaled.shape}")
            except Exception as e:
                print(f"    Warning: Scaling failed: {e}. Using unscaled data.")
                X_scaled = X_np
        else:
            print("    No scaler provided, using unscaled data.")
            X_scaled = X_np


    return X_scaled, scaler

# --- Evaluation Function ---
def evaluate_model(model, X_val, y_val, is_keras=False, batch_size=BATCH_SIZE):
    """Evaluates a model and returns metrics."""
    try:
        if is_keras:
            y_pred_proba = model.predict(X_val, batch_size=batch_size).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'decision_function'):
            # For models like LinearSVC without predict_proba by default
            # (unless using CalibratedClassifierCV)
            y_scores = model.decision_function(X_val)
            y_pred = (y_scores > 0).astype(int)
            # Note: y_scores are not probabilities. For late fusion, ensure probabilities.
            # If using CalibratedClassifierCV, predict_proba will be available.
            if isinstance(model, CalibratedClassifierCV):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                # Cannot directly get probabilities for late fusion without calibration
                print("Warning: Using decision_function scores instead of probabilities for late fusion.")
                y_pred_proba = y_scores # Or apply sigmoid manually: 1 / (1 + np.exp(-y_scores))
        else:
            y_pred = model.predict(X_val)
            y_pred_proba = None # Cannot get probabilities

        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)

        return accuracy, precision, recall, f1, y_pred_proba, y_pred

    except Exception as e:
        print(f"      ERROR during evaluation: {e}")
        return np.nan, np.nan, np.nan, np.nan, None, None


# --- Main Training and Evaluation Loop ---

def train_and_evaluate_fusion_strategies(df, tfidf_sparse_matrix, label_classes):
    """Trains and evaluates different fusion strategies using cross-validation."""

    # --- Define Feature Sets ---
    lexical_features = [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df.columns]
    syntactic_features = [col for col in df.columns if col.startswith('pos_')]
    sentiment_features = [col for col in df.columns if col.startswith('sentiment_')] # Added sentiment
    tfidf_dense_features = [col for col in df.columns if col.startswith('tfidf_')] # Use pre-calculated dense TF-IDF if available

    print("\nAvailable Feature Sets:")
    print(f"- Lexical: {lexical_features}")
    print(f"- Syntactic: {len(syntactic_features)} features (e.g., {syntactic_features[:5]}...)")
    print(f"- Sentiment: {sentiment_features}")
    print(f"- Semantic: ['semantic_vector'] ({SEMANTIC_DIM} dims)")
    if tfidf_sparse_matrix is not None:
        print(f"- TF-IDF (Sparse): Loaded matrix with shape {tfidf_sparse_matrix.shape}")
    if tfidf_dense_features:
        print(f"- TF-IDF (Dense): {len(tfidf_dense_features)} features (e.g., {tfidf_dense_features[:5]}...)")

    # --- Target Variable ---
    y = df['label_encoded'].values

    # --- Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = [] # To store results from each fold and strategy

    # --- Keras Early Stopping ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)

    # --- Cross-Validation Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)): # Split df to easily access columns
        print(f"\n{'='*15} FOLD {fold+1}/{N_SPLITS} {'='*15}")

        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- Prepare Base Feature Sets for this Fold ---
        # TF-IDF (Sparse)
        X_train_tfidf_sparse, X_val_tfidf_sparse = None, None
        if tfidf_sparse_matrix is not None:
            X_train_tfidf_sparse = tfidf_sparse_matrix[train_idx]
            X_val_tfidf_sparse = tfidf_sparse_matrix[val_idx]
            print(f"  Prepared Sparse TF-IDF: Train {X_train_tfidf_sparse.shape}, Val {X_val_tfidf_sparse.shape}")

        # TF-IDF (Dense - if available)
        X_train_tfidf_dense, X_val_tfidf_dense, scaler_tfidf_dense = None, None, None
        if tfidf_dense_features:
            X_train_tfidf_dense, scaler_tfidf_dense = prepare_features(df_train, tfidf_dense_features, fit_scaler=True)
            X_val_tfidf_dense, _ = prepare_features(df_val, tfidf_dense_features, scaler=scaler_tfidf_dense)
            print(f"  Prepared Dense TF-IDF: Train {X_train_tfidf_dense.shape}, Val {X_val_tfidf_dense.shape}")

        # Lexical Features
        X_train_lex, X_val_lex, scaler_lex = None, None, None
        if lexical_features:
            X_train_lex, scaler_lex = prepare_features(df_train, lexical_features, fit_scaler=True)
            X_val_lex, _ = prepare_features(df_val, lexical_features, scaler=scaler_lex)
            print(f"  Prepared Lexical: Train {X_train_lex.shape}, Val {X_val_lex.shape}")

        # Syntactic Features
        X_train_syn, X_val_syn, scaler_syn = None, None, None
        if syntactic_features:
            X_train_syn, scaler_syn = prepare_features(df_train, syntactic_features, fit_scaler=True)
            X_val_syn, _ = prepare_features(df_val, syntactic_features, scaler=scaler_syn)
            print(f"  Prepared Syntactic: Train {X_train_syn.shape}, Val {X_val_syn.shape}")

        # Semantic Features
        X_train_sem, X_val_sem, scaler_sem = None, None, None
        if 'semantic_vector' in df.columns:
            # Semantic vectors are already numpy arrays after preprocessing
            X_train_sem_raw = np.stack(df_train['semantic_vector'].values)
            X_val_sem_raw = np.stack(df_val['semantic_vector'].values)
            # Scale semantic features
            scaler_sem = StandardScaler()
            X_train_sem = scaler_sem.fit_transform(X_train_sem_raw)
            X_val_sem = scaler_sem.transform(X_val_sem_raw)
            print(f"  Prepared Semantic: Train {X_train_sem.shape}, Val {X_val_sem.shape}")

        # --- Strategy 1: Early Fusion (TF-IDF Dense + Lexical + Syntactic -> SVM) ---
        print("\n  --- Strategy: Early Fusion (TF-IDF Dense + Lex + Syn -> SVM) ---")
        if X_train_tfidf_dense is not None and X_train_lex is not None and X_train_syn is not None:
            try:
                # Combine features (ensure all are dense and scaled)
                X_train_early = np.hstack((X_train_tfidf_dense, X_train_lex, X_train_syn))
                X_val_early = np.hstack((X_val_tfidf_dense, X_val_lex, X_val_syn))
                print(f"    Combined Early Features Shape: Train {X_train_early.shape}, Val {X_val_early.shape}")

                # Train SVM (using SVC with RBF kernel as an example, could use LinearSVC too)
                model_early = build_svm(kernel='rbf', probability=True, random_state=RANDOM_STATE + fold)
                print("    Training Early Fusion SVM...")
                model_early.fit(X_train_early, y_train)
                print("    Evaluating Early Fusion SVM...")
                acc, pre, rec, f1, _, _ = evaluate_model(model_early, X_val_early, y_val)
                print(f"    Early Fusion Metrics: Acc={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")
                all_results.append({'fold': fold+1, 'strategy': 'Early (TFIDF_Dense+Lex+Syn -> SVM)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1_score': f1})
                del model_early, X_train_early, X_val_early # Clean up memory
                gc.collect()
            except Exception as e:
                print(f"    ERROR during Early Fusion: {e}")
                all_results.append({'fold': fold+1, 'strategy': 'Early (TFIDF_Dense+Lex+Syn -> SVM)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
        else:
            print("    Skipping Early Fusion: Missing required dense feature sets.")


        # --- Strategy 2: Intermediate Fusion (Semantic + Syntactic -> Bi-LSTM) ---
        print("\n  --- Strategy: Intermediate Fusion (Semantic + Syn -> Bi-LSTM) ---")
        if X_train_sem is not None and X_train_syn is not None:
            try:
                # Ensure syntactic features are scaled
                X_train_syn_scaled, scaler_syn_inter = prepare_features(df_train, syntactic_features, fit_scaler=True)
                X_val_syn_scaled, _ = prepare_features(df_val, syntactic_features, scaler=scaler_syn_inter)

                # Semantic features are already scaled
                current_semantic_dim = X_train_sem.shape[1]
                current_syntactic_dim = X_train_syn_scaled.shape[1]

                model_intermediate = build_bilstm_intermediate(current_semantic_dim, current_syntactic_dim)
                # model_intermediate.summary() # Optional: print model summary

                print("    Training Intermediate Fusion Bi-LSTM...")
                history = model_intermediate.fit(
                    [X_train_sem, X_train_syn_scaled], y_train,
                    validation_data=([X_val_sem, X_val_syn_scaled], y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping],
                    verbose=0 # Set to 1 or 2 for more details during training
                )
                print(f"    Training completed after {len(history.history['loss'])} epochs.")

                print("    Evaluating Intermediate Fusion Bi-LSTM...")
                acc, pre, rec, f1, _, _ = evaluate_model(model_intermediate, [X_val_sem, X_val_syn_scaled], y_val, is_keras=True)
                print(f"    Intermediate Fusion Metrics: Acc={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")
                all_results.append({'fold': fold+1, 'strategy': 'Intermediate (Sem+Syn -> BiLSTM)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1_score': f1})

                tf.keras.backend.clear_session() # Clear Keras session
                del model_intermediate, X_train_syn_scaled, X_val_syn_scaled, scaler_syn_inter, history # Clean up memory
                gc.collect()

            except Exception as e:
                print(f"    ERROR during Intermediate Fusion: {e}")
                all_results.append({'fold': fold+1, 'strategy': 'Intermediate (Sem+Syn -> BiLSTM)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
                tf.keras.backend.clear_session() # Ensure session is cleared even on error
                gc.collect()
        else:
            print("    Skipping Intermediate Fusion: Missing Semantic or Syntactic features.")


        # --- Strategy 3: Late Fusion (Average Probabilities: TF-IDF Sparse SVM + Semantic Dense NN) ---
        print("\n  --- Strategy: Late Fusion (TF-IDF_Sparse->SVM + Semantic->DenseNN) ---")
        if X_train_tfidf_sparse is not None and X_train_sem is not None:
            try:
                # Model 1: SVM on TF-IDF Sparse
                print("    Training Late Fusion Model 1 (SVM on TF-IDF)...")
                # Use LinearSVC with calibration for probabilities
                model_late_svm = build_svm(kernel='linear', probability=True, random_state=RANDOM_STATE + fold)
                model_late_svm.fit(X_train_tfidf_sparse, y_train)
                print("    Evaluating Late Fusion Model 1 (SVM)...")
                _, _, _, _, svm_probs, _ = evaluate_model(model_late_svm, X_val_tfidf_sparse, y_val)

                # Model 2: Dense NN on Semantic Features
                print("    Training Late Fusion Model 2 (Dense NN on Semantic)...")
                tf.random.set_seed(RANDOM_STATE + fold) # Seed for Keras model
                model_late_nn = build_dense_for_semantic(input_dim=X_train_sem.shape[1])
                history_nn = model_late_nn.fit(X_train_sem, y_train,
                                               epochs=EPOCHS,
                                               batch_size=BATCH_SIZE,
                                               validation_data=(X_val_sem, y_val),
                                               callbacks=[early_stopping],
                                               verbose=0)
                print(f"    NN Training completed after {len(history_nn.history['loss'])} epochs.")
                print("    Evaluating Late Fusion Model 2 (NN)...")
                _, _, _, _, nn_probs, _ = evaluate_model(model_late_nn, X_val_sem, y_val, is_keras=True)

                # Late Fusion: Average Probabilities
                if svm_probs is not None and nn_probs is not None:
                    avg_probs = (svm_probs + nn_probs) / 2
                    y_pred_late = (avg_probs > 0.5).astype(int)

                    # Evaluate Late Fusion
                    accuracy_late = accuracy_score(y_val, y_pred_late)
                    precision_late, recall_late, f1_late, _ = precision_recall_fscore_support(y_val, y_pred_late, average='binary', zero_division=0)
                    print(f"    Late Fusion Metrics: Acc={accuracy_late:.4f}, P={precision_late:.4f}, R={recall_late:.4f}, F1={f1_late:.4f}")
                    all_results.append({'fold': fold+1, 'strategy': 'Late Fusion (Avg Prob)', 'accuracy': accuracy_late, 'precision': precision_late, 'recall': recall_late, 'f1_score': f1_late})
                else:
                    print("    Skipping Late Fusion evaluation due to missing probabilities.")
                    all_results.append({'fold': fold+1, 'strategy': 'Late Fusion (Avg Prob)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})


                tf.keras.backend.clear_session() # Clear Keras session
                del model_late_svm, model_late_nn, svm_probs, nn_probs, history_nn # Clean up memory
                gc.collect()

            except Exception as e:
                print(f"    ERROR during Late Fusion: {e}")
                all_results.append({'fold': fold+1, 'strategy': 'Late Fusion (Avg Prob)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
                tf.keras.backend.clear_session() # Ensure session is cleared even on error
                gc.collect()
        else:
            print("    Skipping Late Fusion: Missing Sparse TF-IDF or Semantic features.")

        # --- Optional: Add Baseline Models (from Phase 3 logic if needed for comparison) ---
        # Example: Baseline SVM on TF-IDF only
        print("\n  --- Strategy: Baseline (TF-IDF Sparse -> SVM) ---")
        if X_train_tfidf_sparse is not None:
            try:
                model_base_svm = build_svm(kernel='linear', probability=False, random_state=RANDOM_STATE + fold) # No need for probability here
                print("    Training Baseline SVM...")
                model_base_svm.fit(X_train_tfidf_sparse, y_train)
                print("    Evaluating Baseline SVM...")
                acc, pre, rec, f1, _, _ = evaluate_model(model_base_svm, X_val_tfidf_sparse, y_val)
                print(f"    Baseline TF-IDF SVM Metrics: Acc={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")
                all_results.append({'fold': fold+1, 'strategy': 'Baseline (TF-IDF Sparse -> SVM)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1_score': f1})
                del model_base_svm
                gc.collect()
            except Exception as e:
                print(f"    ERROR during Baseline SVM: {e}")
                all_results.append({'fold': fold+1, 'strategy': 'Baseline (TF-IDF Sparse -> SVM)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
        else:
            print("    Skipping Baseline TF-IDF SVM: Missing Sparse TF-IDF features.")

        # Example: Baseline Dense NN on Semantic only
        print("\n  --- Strategy: Baseline (Semantic -> Dense NN) ---")
        if X_train_sem is not None:
            try:
                tf.random.set_seed(RANDOM_STATE + fold) # Seed for Keras model
                model_base_nn = build_dense_for_semantic(input_dim=X_train_sem.shape[1])
                print("    Training Baseline Dense NN...")
                history_base_nn = model_base_nn.fit(X_train_sem, y_train,
                                                    epochs=EPOCHS,
                                                    batch_size=BATCH_SIZE,
                                                    validation_data=(X_val_sem, y_val),
                                                    callbacks=[early_stopping],
                                                    verbose=0)
                print(f"    Training completed after {len(history_base_nn.history['loss'])} epochs.")
                print("    Evaluating Baseline Dense NN...")
                acc, pre, rec, f1, _, _ = evaluate_model(model_base_nn, X_val_sem, y_val, is_keras=True)
                print(f"    Baseline Semantic NN Metrics: Acc={acc:.4f}, P={pre:.4f}, R={rec:.4f}, F1={f1:.4f}")
                all_results.append({'fold': fold+1, 'strategy': 'Baseline (Semantic -> Dense NN)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1_score': f1})

                tf.keras.backend.clear_session() # Clear Keras session
                del model_base_nn, history_base_nn # Clean up memory
                gc.collect()
            except Exception as e:
                print(f"    ERROR during Baseline Dense NN: {e}")
                all_results.append({'fold': fold+1, 'strategy': 'Baseline (Semantic -> Dense NN)', 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan})
                tf.keras.backend.clear_session() # Ensure session is cleared even on error
                gc.collect()
        else:
            print("    Skipping Baseline Semantic NN: Missing Semantic features.")

        # Clean up common variables for the fold
        del df_train, df_val, y_train, y_val
        del X_train_tfidf_sparse, X_val_tfidf_sparse
        if tfidf_dense_features: del X_train_tfidf_dense, X_val_tfidf_dense, scaler_tfidf_dense
        if lexical_features: del X_train_lex, X_val_lex, scaler_lex
        if syntactic_features: del X_train_syn, X_val_syn, scaler_syn
        if 'semantic_vector' in df.columns: del X_train_sem, X_val_sem, scaler_sem, X_train_sem_raw, X_val_sem_raw
        gc.collect()


    # --- Final Results Processing ---
    results_df = pd.DataFrame(all_results)
    return results_df

# --- Main Execution ---
if __name__ == "__main__":
    # Define paths
    data_dir = Path('data/processed')
    input_file = data_dir / 'phase2_output.csv'
    tfidf_sparse_file = data_dir / 'tfidf_matrix.npz' # Assuming this was saved in phase 1 or 2
    output_dir = Path('results/phase4') # Specific output dir for this phase
    output_file = output_dir / 'fusion_model_results.csv'
    summary_file = output_dir / 'fusion_model_summary.csv'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df, tfidf_sparse_matrix, label_classes = load_and_preprocess_data(input_file, tfidf_sparse_file)

    # Check if DataFrame is empty after preprocessing
    if df.empty:
        print("Error: DataFrame is empty after preprocessing. Cannot proceed with training.")
        sys.exit(1)

    # Train and evaluate models with different fusion strategies
    results_df = train_and_evaluate_fusion_strategies(df, tfidf_sparse_matrix, label_classes)

    # Save detailed results
    if not results_df.empty:
        print(f"\nSaving detailed fold results to: {output_file}")
        results_df.to_csv(output_file, index=False)

        # Calculate and display average results per strategy
        print("\n--- Average Performance Across Folds (Grouped by Strategy) ---")
        avg_results = results_df.groupby('strategy')[['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean', 'std'])

        # Format for better readability
        avg_results.columns = ['_'.join(col).strip() for col in avg_results.columns.values]
        avg_results = avg_results.rename(columns={
            'accuracy_mean': 'Avg Accuracy', 'accuracy_std': 'Std Dev Acc',
            'precision_mean': 'Avg Precision', 'precision_std': 'Std Dev Prec',
            'recall_mean': 'Avg Recall', 'recall_std': 'Std Dev Rec',
            'f1_score_mean': 'Avg F1-Score', 'f1_score_std': 'Std Dev F1'
        })
        avg_results = avg_results.sort_values(by='Avg F1-Score', ascending=False)

        print(avg_results)
        print(f"\nSaving summary results to: {summary_file}")
        avg_results.to_csv(summary_file)

    else:
        print("\nNo results were generated during model training and evaluation.")

    print("\nPhase 4 Processing complete.")


# ----- Previous Feature Fusion Functions -----
# def early_fusion(df):
#     features = ['tfidf_feature1', 'tfidf_feature2',
#                 'word_count', 'pos_NOUN', 'semantic_vector']
#     X = df[features]
#     X_train, X_test, y_train, y_test = train_test_split(X, df['label'])
#     model = SVC().fit(X_train, y_train)
#     return f1_score(y_test, model.predict(X_test))
#
# def late_fusion(svm_probs, lstm_probs):
#     avg_probs = (svm_probs + lstm_probs) / 2
#     return (avg_probs > 0.5).astype(int)

# # Example usage
# # svm_model = SVC().fit(tfidf_features, labels)
# # lstm_model = build_lstm().fit(semantic_features, labels)
# #
# # svm_probs = svm_model.predict_proba(tfidf_test)[:, 1]
# # lstm_probs = lstm_model.predict(semantic_test)
# # final_preds = late_fusion(svm_probs, lstm_probs)
#
# if __name__ == "__main__":
#     data_dir = Path('data/processed')
#     input_file = data_dir / 'phase2_output.csv'
#     output_dir = Path('results')
#
#     # load features
#     df = pd.read_csv(input_file)
#     # separate features and labels
#     features = ['tfidf_feature1', 'tfidf_feature2',
#                 'word_count', 'pos_NOUN', 'semantic_vector']
#     X = df[features]
#     y = df['label']
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     # Train SVM model
#     svm_model = SVC(probability=True)
#
#     svm_model.fit(X_train, y_train)
#     # Get probabilities
#     svm_probs = svm_model.predict_proba(X_test)[:, 1]
#     # Assuming lstm_model is defined and trained
#     lstm_model = build_lstm().fit(semantic_features, labels)
#
#     # Example DataFrame
#     data = {
#         'tfidf_feature1': [0.1, 0.2, 0.3],
#         'tfidf_feature2': [0.4, 0.5, 0.6],
#         'word_count': [100, 150, 200],
#         'pos_NOUN': [10, 15, 20],
#         'semantic_vector': [[0.1]*384, [0.2]*384, [0.3]*384],
#         'label': [0, 1, 0]
#     }
#     df = pd.DataFrame(data)
#     print(early_fusion(df))