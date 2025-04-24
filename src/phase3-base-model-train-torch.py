import ast  # For safely evaluating string representations of lists
import gc  # Garbage collector
import os
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# --- Constants ---
N_SPLITS = 5  # Number of folds for cross-validation on the training set
RANDOM_STATE = 42
SEMANTIC_DIM = 384  # Dimension of 'all-MiniLM-L6-v2' embeddings (will be updated if different)
EPOCHS = 50  # Max epochs for PyTorch models
BATCH_SIZE = 32
PATIENCE = 5  # Early stopping patience
TEST_SET_SIZE = 0.2  # Proportion of data to hold out for the final test set
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available


# --- Model Building Functions ---

def build_svm(kernel='rbf', probability=False, class_weight='balanced', random_state=None):
    """Builds a Support Vector Classifier."""
    if kernel == 'linear':
        return LinearSVC(
            class_weight=class_weight,
            dual='auto',
            random_state=random_state,
            max_iter=5000,
            C=1.0,
            tol=1e-3
        )
    else:  # RBF kernel
        return SVC(kernel=kernel, probability=probability, class_weight=class_weight, random_state=random_state)


class CNN1D(nn.Module):
    """Builds a 1D CNN model for flat feature vectors (e.g., syntactic)."""

    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=input_dim - 4)  # Ensure pooling to 1 element
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten for dense layer
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class DenseNetwork(nn.Module):
    """Builds a simple Dense network for semantic vectors."""

    def __init__(self, input_dim):
        super(DenseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# --- Data Loading and Preprocessing ---

def safe_literal_eval(x):
    """Safely evaluates a string literal, returning an empty array on error."""
    try:
        if isinstance(x, str):
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.float32)
        return np.array([], dtype=np.float32)
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)


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

    # 2. Parse 'semantic_vector'
    global SEMANTIC_DIM  # Allow modification of the global constant
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
            if first_valid_vector_dim != SEMANTIC_DIM:
                print(
                    f"Warning: Expected semantic vector dimension {SEMANTIC_DIM}, but found {first_valid_vector_dim}. Using actual dimension: {first_valid_vector_dim}")
                SEMANTIC_DIM = first_valid_vector_dim  # Update global constant
            inconsistent_dims = df['semantic_vector'].apply(lambda x: x.shape[0] != SEMANTIC_DIM).sum()
            if inconsistent_dims > 0:
                print(f"Error: Found {inconsistent_dims} semantic vectors with inconsistent dimensions.")
                sys.exit(1)
    else:
        print("Warning: 'semantic_vector' column not found. Semantic features will be skipped.")

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

    # 6. Handle TF-IDF features (check for dense columns, assume sparse is separate)
    tfidf_sparse_path = Path('data/processed/tfidf_matrix.npz')
    tfidf_sparse_matrix = None
    if not tfidf_sparse_path.exists():
        print(
            f"Warning: Sparse TF-IDF matrix file not found at {tfidf_sparse_path}. 'tfidf_sparse' group will be skipped.")
    else:
        try:
            tfidf_sparse_matrix = load_npz(tfidf_sparse_path)
            print(f"Loaded sparse TF-IDF matrix with shape: {tfidf_sparse_matrix.shape}")
            # Adjust DataFrame rows to match sparse matrix if necessary and possible
            if tfidf_sparse_matrix.shape[0] != len(df):
                print(
                    f"Warning: Row count mismatch between data ({len(df)}) and sparse TF-IDF ({tfidf_sparse_matrix.shape[0]}). Attempting to align based on index if possible or exiting.")
                # This assumes the sparse matrix corresponds to the original unfiltered data index
                # If df was filtered *before* this point, alignment might fail.
                if len(df) > tfidf_sparse_matrix.shape[0]:
                    # Example: Try aligning if df has extra rows (e.g., due to less filtering than when matrix was created)
                    # This is risky and depends heavily on how the matrix and df were created.
                    # A safer approach is to ensure they are generated consistently.
                    print("Attempting to filter DataFrame to match sparse matrix rows. This is experimental.")
                    # Assuming df.index aligns with original data before filtering
                    if df.index.max() >= tfidf_sparse_matrix.shape[0]:
                        df = df[df.index < tfidf_sparse_matrix.shape[0]]
                        print(f"Filtered DataFrame shape: {df.shape}")
                    else:
                        print("Cannot align DataFrame index with sparse matrix shape.")
                        sys.exit(1)

                if tfidf_sparse_matrix.shape[0] != len(df):
                    print("Row count mismatch persists after alignment attempt. Exiting.")
                    sys.exit(1)
                else:
                    print("DataFrame aligned with sparse matrix.")

        except Exception as e:
            print(f"Error loading sparse TF-IDF matrix: {e}")
            tfidf_sparse_matrix = None  # Ensure it's None if loading fails

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


# --- Feature Preparation Helper ---

def prepare_features(df, feature_cols, scaler=None, fit_scaler=False, is_sparse=False):
    """Extracts, handles NaNs, and optionally scales features."""
    if is_sparse:
        if df is None:
            raise ValueError("Sparse matrix not provided or loaded correctly.")
        return df, None  # No scaler for sparse

    # For dense features
    X = df[feature_cols].copy()
    if X.isnull().values.any():
        print(f"  Warning: NaN values found in features {feature_cols}. Imputing with 0.")
        X.fillna(0, inplace=True)
    X_np = X.values.astype(np.float32)

    current_scaler = scaler if scaler is not None else StandardScaler()

    if fit_scaler:
        X_scaled = current_scaler.fit_transform(X_np)
        return X_scaled, current_scaler
    else:
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_np)
            except Exception as e:
                print(f"    Warning: Scaling failed: {e}. Using unscaled data.")
                X_scaled = X_np
        else:
            X_scaled = X_np
        return X_scaled, scaler


# --- Evaluation Function ---

def evaluate_model(model, X_val, y_val, is_pytorch=False, batch_size=BATCH_SIZE):
    """Evaluates a model and returns metrics."""

    try:
        y_pred_proba = None
        y_pred = None
        if is_pytorch:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation
                X_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                y_pred_proba = model(X_tensor).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_val)
            y_pred = (y_scores > 0).astype(int)
            # Probabilities might not be available for LinearSVC unless calibrated
        else:
            y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)
        return accuracy, precision, recall, f1, y_pred_proba, y_pred
    except Exception as e:
        print(f"      ERROR during evaluation: {e}")
        return np.nan, np.nan, np.nan, np.nan, None, None


# --- Cross-Validation Function ---

def cross_validate_models(df_train, tfidf_sparse_matrix_train, label_classes):
    """Performs cross-validation on the training set for each feature group."""

    print("\n===== Starting Cross-Validation on Training Set =====")

    # Define feature groups based on training data columns
    feature_groups = {
        'lexical': [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df_train.columns],
        'syntactic': [col for col in df_train.columns if col.startswith('pos_')],
        'semantic': ['semantic_vector'] if 'semantic_vector' in df_train.columns else [],
        'tfidf_sparse': ['tfidf_matrix.npz'] if tfidf_sparse_matrix_train is not None else [],
        'tfidf_dense': [col for col in df_train.columns if col.startswith('tfidf_')],
    }
    feature_groups = {name: cols for name, cols in feature_groups.items() if cols}  # Filter empty groups

    print("\nFeature groups for CV:")
    for name, cols in feature_groups.items():
        if name == 'tfidf_sparse':
            print(f"- {name}: Using provided sparse matrix")
        elif name == 'semantic':
            print(f"- {name}: {SEMANTIC_DIM} features from '{cols[0]}'")
        else:
            print(f"- {name}: {len(cols)} features")

    # Target variable from training data
    y_train_full = df_train['label_encoded'].values

    # Cross-Validation setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_cv_results = []  # Store results from each fold, model, and group

    # PyTorch early stopping callback (manual implementation)
    def early_stopping(val_loss_history, patience):
        if len(val_loss_history) < patience:
            return False
        best_loss = min(val_loss_history)
        last_losses = val_loss_history[-patience:]
        return all(loss > best_loss for loss in last_losses)

    early_stopping = early_stopping(val_loss_history=[], patience=PATIENCE)

    # Iterate over each feature group
    # --- Iterate through defined feature groups ---
    for group_name, feature_cols in feature_groups.items():
        print(f"\n{'=' * 10} CV for Feature Group: {group_name} {'=' * 10}")

        is_pytorch_model_group = group_name in ['syntactic', 'semantic']
        is_sklearn_group = group_name in ['lexical', 'tfidf_sparse', 'tfidf_dense']

        # --- Prepare data (X) for the current group using ONLY training data indices ---
        X_train_group = None
        input_dim = 0
        is_sparse = False

        try:
            if group_name == 'semantic':
                X_train_group = np.stack(df_train[feature_cols[0]].values)
                input_dim = X_train_group.shape[1]
            elif group_name == 'tfidf_sparse':
                X_train_group = tfidf_sparse_matrix_train  # Use the pre-split training sparse matrix
                input_dim = X_train_group.shape[1]
                is_sparse = True
            elif group_name in ['lexical', 'syntactic', 'tfidf_dense']:
                X_train_group = df_train[feature_cols].values
                if np.isnan(X_train_group).any():
                    X_train_group = np.nan_to_num(X_train_group)
                input_dim = X_train_group.shape[1]
            else:
                continue  # Should not happen

            if X_train_group is None or input_dim == 0:
                continue
            if X_train_group.shape[0] != len(y_train_full):
                continue

        except Exception as e:
            print(f"  Error preparing CV data for group '{group_name}': {e}")
            continue

        # --- Cross-Validation Loop for this group ---
        fold_results_group = defaultdict(lambda: defaultdict(list))

        for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_train_group, y_train_full)):
            print(f"\n  --- CV Fold {fold + 1}/{N_SPLITS} ---")

            # --- Data Splitting for Fold ---
            if is_sparse:
                X_train_fold, X_val_fold = X_train_group[train_fold_idx], X_train_group[val_fold_idx]
            else:
                X_train_fold, X_val_fold = X_train_group[train_fold_idx].astype(np.float32), X_train_group[
                    val_fold_idx].astype(np.float32)
            y_train_fold, y_val_fold = y_train_full[train_fold_idx], y_train_full[val_fold_idx]

            # --- Scaling (Fit on Train Fold, Transform Val Fold) ---
            scaler = None
            if not is_sparse:
                scaler = StandardScaler()
                try:
                    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                    X_val_fold_scaled = scaler.transform(X_val_fold)
                except ValueError as e:
                    print(f"    Warning: Scaling failed for fold {fold + 1}: {e}. Using unscaled data.")
                    X_train_fold_scaled = X_train_fold
                    X_val_fold_scaled = X_val_fold
            else:
                X_train_fold_scaled = X_train_fold
                X_val_fold_scaled = X_val_fold

            # --- Define Models for this Fold ---
            current_models_to_run = {}
            fold_random_state = RANDOM_STATE + fold

            if is_sklearn_group:
                count_neg = np.sum(y_train_fold == 0)
                count_pos = np.sum(y_train_fold == 1)
                scale_pos_weight_val = count_neg / count_pos if count_pos > 0 else 1
                sklearn_models = {
                    'LogisticRegression': LogisticRegression(penalty='l2', C=1.0, solver='liblinear',
                                                             class_weight='balanced', random_state=fold_random_state,
                                                             max_iter=2000),
                    'LinearSVC': LinearSVC(class_weight='balanced', dual='auto', random_state=fold_random_state,
                                           max_iter=5000, C=1.0, tol=1e-3),
                    'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=None,
                                                           class_weight='balanced_subsample',
                                                           random_state=fold_random_state, n_jobs=-1,
                                                           min_samples_split=5),
                    'ComplementNB': ComplementNB(),
                    'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                             use_label_encoder=False, tree_method='hist',
                                             scale_pos_weight=scale_pos_weight_val,
                                             random_state=fold_random_state, n_estimators=100)
                }
                if group_name == 'lexical':
                    sklearn_models['SVC_RBF'] = SVC(kernel='rbf', probability=False, class_weight='balanced',
                                                    random_state=fold_random_state)  # Probability False for CV speed
                current_models_to_run.update(sklearn_models)

            elif is_pytorch_model_group:
                torch.manual_seed(fold_random_state)
                if group_name == 'syntactic':
                    model = CNN1D(input_dim=input_dim).to(DEVICE)
                    current_models_to_run[model.__class__.__name__] = model
                elif group_name == 'semantic':
                    model = DenseNetwork(input_dim=input_dim).to(DEVICE)
                    current_models_to_run[model.__class__.__name__] = model

            # Initialize variables to track the best model
            best_model_f1 = None
            best_model = None
            best_model_weights = None
            best_model_name = None
            best_model_group = None
            best_model_fold = None
            best_model_epoch = None
            best_model_y_pred = None
            best_model_y_pred_proba = None
            best_model_y_true = None
            best_model_accuracy = None
            best_model_precision = None
            best_model_recall = None
            best_model_auc = None

            # --- Train and Evaluate each model for the current fold ---
            for model_name, model_instance in current_models_to_run.items():

                current_X_train = X_train_fold_scaled
                current_X_val = X_val_fold_scaled
                model_requires_dense = model_name in ['RandomForest', 'XGBoost', 'SVC_RBF'] or is_pytorch_model_group

                if is_sparse and model_requires_dense:
                    try:
                        current_X_train = current_X_train.toarray()
                        current_X_val = current_X_val.toarray()
                    except MemoryError:
                        continue  # Skip if conversion fails
                    except Exception:
                        continue

                # --- Training & Evaluation ---
                try:
                    if is_pytorch_model_group and model_name == model_instance.__class__.__name__:
                        # Convert data to PyTorch tensors
                        X_train_tensor = torch.tensor(current_X_train, dtype=torch.float32).to(DEVICE)
                        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).to(DEVICE)
                        X_val_tensor = torch.tensor(current_X_val, dtype=torch.float32).to(DEVICE)
                        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(DEVICE)

                        # Create DataLoader
                        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
                        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                        # Define loss function and optimizer
                        criterion = nn.BCELoss()
                        optimizer = optim.Adam(model_instance.parameters(), lr=0.0005)

                        # Training loop with early stopping
                        val_loss_history = []  # Store validation loss for early stopping
                        for epoch in range(EPOCHS):
                            model_instance.train()  # Set model to training mode
                            epoch_loss = 0.0
                            for X_batch, y_batch in train_loader:
                                optimizer.zero_grad()  # Zero the parameter gradients
                                y_pred = model_instance(X_batch)  # Forward pass
                                loss = criterion(y_pred, y_batch)  # Calculate loss
                                loss.backward()  # Backward pass
                                optimizer.step()  # Optimize
                                epoch_loss += loss.item() * X_batch.size(0)
                            epoch_loss /= len(train_loader.dataset)

                            # Evaluate on validation set
                            model_instance.eval()  # Set model to evaluation mode
                            with torch.no_grad():
                                val_pred = model_instance(X_val_tensor)
                                val_loss = criterion(val_pred, y_val_tensor.unsqueeze(1)).item()

                            val_loss_history.append(val_loss)
                            #print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

                            # Early stopping check
                            if early_stopping(val_loss_history, PATIENCE):
                                print(f"Early stopping triggered after {epoch + 1} epochs.")
                                break
                        acc, pre, rec, f1, y_pred_proba, y_pred = evaluate_model(model_instance, current_X_val,
                                                                                 y_val_fold, is_pytorch=True)
                        #clear the memory after training and evaluation
                        del model_instance, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, train_dataset, train_loader, criterion, optimizer, val_loss_history, epoch_loss, X_batch, y_batch, y_pred, loss, val_pred, val_loss
                    else:  # Scikit-learn models
                        model_instance.fit(current_X_train, y_train_fold)
                        acc, pre, rec, f1, y_pred_proba, y_pred = evaluate_model(model_instance, current_X_val,
                                                                                 y_val_fold)
                        del model_instance

                    #check if the current is better than the best model
                    if f1 > best_model_f1 or best_model_f1 is None:
                        best_model_f1 = f1
                        best_model = model_instance
                        best_model_weights = model_instance.state_dict()
                        best_model_name = model_name
                        best_model_group = group_name
                        best_model_fold = fold
                        best_model_epoch = epoch
                        best_model_y_pred = y_pred
                        best_model_y_pred_proba = y_pred_proba
                        best_model_y_true = y_val_fold
                        best_model_accuracy = acc
                        best_model_precision = pre
                        best_model_recall = rec
                        best_model_auc = 0
                        print(f"current best model, group: {best_model_group}, model name: {best_model_name}, fold: {best_model_fold}, epoch: {best_model_epoch}, f1: {best_model_f1}")

                except Exception as e:
                    print(f"      ERROR during CV training/prediction for {model_name}: {e}")
                    acc, pre, rec, f1, y_pred_proba, y_pred = np.nan, np.nan, np.nan, np.nan, None, None

                # Store results
                fold_results_group[model_name]['fold'].append(fold + 1)
                fold_results_group[model_name]['accuracy'].append(acc)
                fold_results_group[model_name]['precision'].append(pre)
                fold_results_group[model_name]['recall'].append(rec)
                fold_results_group[model_name]['f1_score'].append(f1)

            # Clean up fold data
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold
            del X_train_fold_scaled, X_val_fold_scaled, scaler
            gc.collect()

        # --- Aggregate CV results for the feature group ---
        print(f"\n  --- Aggregated CV Results for Feature Group: {group_name} ---")
        for model_name, metrics_dict in fold_results_group.items():
            avg_f1 = np.nanmean(metrics_dict['f1_score'])
            print(f"    Model: {model_name:<20} Avg F1: {avg_f1:.4f}")
            # Append all fold results to the main list
            for i in range(len(metrics_dict['fold'])):
                all_cv_results.append({
                    'feature_group': group_name,
                    'model_name': model_name,
                    'fold': metrics_dict['fold'][i],
                    'accuracy': metrics_dict['accuracy'][i],
                    'precision': metrics_dict['precision'][i],
                    'recall': metrics_dict['recall'][i],
                    'f1_score': metrics_dict['f1_score'][i]
                })
        del X_train_group  # Clean up group data
        gc.collect()
        # --- End of cross-validation for this group ---
        print(f"\n{'=' * 10} Finished CV for Feature Group: {group_name} {'=' * 10}")
        print("\n===== Finished Cross-Validation on Training Set =====")

        # --- Save CV results to CSV ---
        cv_results_df = pd.DataFrame(all_cv_results)
        cv_results_df.to_csv('cv_results.csv', index=False)
        print(f"Cross-validation results saved to cv_results.csv")
        return cv_results_df


# --- Function to Save Final Model (Modified for phase 3 context) ---
def save_final_model(model, feature_group, model_name, output_dir):
    """Saves the final model and its configuration."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, f"{feature_group}_{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Final model saved to: {model_path}")
    return model_path


# --- Main Execution (Modified for Train/CV/Test Workflow) ---
if __name__ == "__main__":
    # Define paths
    data_dir = Path('data/processed')
    input_file = data_dir / 'phase2_output.csv'
    output_dir = Path('results/phase3')
    cv_output_file = output_dir / 'cv_model_training_results.csv'
    cv_summary_file = output_dir / 'cv_model_training_summary.csv'
    test_output_file = output_dir / 'test_set_results.csv' # New file for test results
    final_model_dir = Path('models/phase3') # Directory to save final models

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    df, tfidf_sparse_matrix, label_classes = load_and_preprocess_data(input_file)
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
    df_train = df.iloc[train_indices].copy().reset_index(drop=True)
    df_test = df.iloc[test_indices].copy().reset_index(drop=True)
    y_train_full = df_train['label_encoded'].values
    y_test = df_test['label_encoded'].values

    tfidf_sparse_matrix_train = None
    tfidf_sparse_matrix_test = None
    if tfidf_sparse_matrix is not None:
        try:
            tfidf_sparse_matrix_train = tfidf_sparse_matrix[train_indices]
            tfidf_sparse_matrix_test = tfidf_sparse_matrix[test_indices]
            print(f"Split sparse TF-IDF: Train {tfidf_sparse_matrix_train.shape}, Test {tfidf_sparse_matrix_test.shape}")
        except IndexError as e:
            print(f"Error splitting sparse matrix, likely due to index mismatch after preprocessing: {e}")
            print("Skipping sparse TF-IDF features.")
            tfidf_sparse_matrix_train = None # Ensure it's None if split fails
            tfidf_sparse_matrix_test = None


    print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")

    # 3. Perform Cross-Validation on the Training Set
    cv_results_df = cross_validate_models(df_train, tfidf_sparse_matrix_train, label_classes)

    # 4. Save and Analyze CV Results
    best_models_from_cv = {} # Store best model name for each group: {group_name: model_name}
    if not cv_results_df.empty:
        print(f"\nSaving detailed CV results to: {cv_output_file}")
        cv_results_df.to_csv(cv_output_file, index=False)

        print("\n--- Average CV Performance Across Folds (Grouped by Feature Group & Model) ---")
        avg_cv_results = cv_results_df.groupby(['feature_group', 'model_name'])[['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean', 'std'])
        avg_cv_results.columns = ['_'.join(col).strip() for col in avg_cv_results.columns.values]
        avg_cv_results = avg_cv_results.rename(columns={
            'accuracy_mean': 'Avg Accuracy', 'accuracy_std': 'Std Dev Acc',
            'precision_mean': 'Avg Precision', 'precision_std': 'Std Dev Prec',
            'recall_mean': 'Avg Recall', 'recall_std': 'Std Dev Rec',
            'f1_score_mean': 'Avg F1-Score', 'f1_score_std': 'Std Dev F1'
        })
        # Sort within each group to find the best model
        avg_cv_results = avg_cv_results.sort_values(by=['feature_group', 'Avg F1-Score'], ascending=[True, False])
        print(avg_cv_results)
        print(f"\nSaving CV summary results to: {cv_summary_file}")
        avg_cv_results.to_csv(cv_summary_file)

        # Identify best model name for each feature group based on Avg F1-Score
        for group in avg_cv_results.index.get_level_values('feature_group').unique():
            best_model_entry = avg_cv_results.loc[group].iloc[0] # First entry is best due to sorting
            best_models_from_cv[group] = best_model_entry.name # Get the model name (index)
        print(f"\nBest models selected from CV: {best_models_from_cv}")

    else:
        print("\nNo CV results were generated. Cannot determine best models or proceed.")
        sys.exit(1)

    # 5. Train Final Models (Best from CV) on Full Training Set and Evaluate on Test Set
    print(f"\n===== Training Final Models on Full Training Set & Evaluating on Test Set =====")
    all_test_results = []
    saved_final_models = {} # Store paths of saved final models
    final_scalers = {} # Store scalers fitted on full training data

    # Define feature sets based on full training data
    feature_groups_final = {
        'lexical': [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df_train.columns],
        'syntactic': [col for col in df_train.columns if col.startswith('pos_')],
        'semantic': ['semantic_vector'] if 'semantic_vector' in df_train.columns else [],
        'tfidf_sparse': ['tfidf_matrix.npz'] if tfidf_sparse_matrix_train is not None else [],
        'tfidf_dense': [col for col in df_train.columns if col.startswith('tfidf_')],
    }
    feature_groups_final = {name: cols for name, cols in feature_groups_final.items() if cols} # Filter empty

    # Fit scalers on FULL training data ONCE
    print("\nFitting scalers on full training data...")
    for group_name, feature_cols in feature_groups_final.items():
        if group_name not in ['tfidf_sparse', 'semantic']: # Scale dense features
            _, scaler = prepare_features(df_train, feature_cols, fit_scaler=True)
            if scaler:
                final_scalers[group_name] = scaler
                joblib.dump(scaler, final_model_dir / f'scaler_{group_name}.pkl') # Save scaler
        elif group_name == 'semantic':
            X_train_sem_raw_full = np.stack(df_train[feature_cols[0]].values)
            scaler = StandardScaler().fit(X_train_sem_raw_full)
            final_scalers[group_name] = scaler
            joblib.dump(scaler, final_model_dir / f'scaler_{group_name}.pkl') # Save scaler
    print("Saved scalers fitted on full training data.")


    # Iterate through feature groups and train the best model identified from CV
    for group_name, best_model_name in best_models_from_cv.items():
        print(f"\n--- Processing Final Model for Group: {group_name} (Best Model: {best_model_name}) ---")

        feature_cols = feature_groups_final[group_name]
        final_model = None
        test_features = None
        is_keras_final = False
        is_sparse_final = (group_name == 'tfidf_sparse')
        acc_test, pre_test, rec_test, f1_test = np.nan, np.nan, np.nan, np.nan

        try:
            # --- Prepare Features for Train (Full) and Test ---
            X_train_final_scaled, _ = None, None
            X_test_scaled, _ = None, None
            input_dim_final = 0

            if is_sparse_final:
                X_train_final_scaled = tfidf_sparse_matrix_train
                test_features = tfidf_sparse_matrix_test
                if X_train_final_scaled is None or test_features is None: raise ValueError("Sparse matrix missing")
                input_dim_final = X_train_final_scaled.shape[1]
            elif group_name == 'semantic':
                X_train_sem_raw = np.stack(df_train[feature_cols[0]].values)
                X_test_sem_raw = np.stack(df_test[feature_cols[0]].values)
                scaler = final_scalers.get(group_name)
                if not scaler: raise ValueError(f"Scaler not found for {group_name}")
                X_train_final_scaled = scaler.transform(X_train_sem_raw)
                test_features = scaler.transform(X_test_sem_raw)
                input_dim_final = X_train_final_scaled.shape[1]
            else: # Dense sklearn groups
                X_train_final_scaled, _ = prepare_features(df_train, feature_cols, scaler=final_scalers.get(group_name), fit_scaler=False)
                test_features, _ = prepare_features(df_test, feature_cols, scaler=final_scalers.get(group_name), fit_scaler=False)
                if X_train_final_scaled is None or test_features is None: raise ValueError("Feature preparation failed")
                input_dim_final = X_train_final_scaled.shape[1]

            # --- Instantiate and Train Final Model ---
            print(f"  Instantiating and training final {best_model_name} model...")
            model_requires_dense = best_model_name in ['RandomForest', 'XGBoost', 'SVC_RBF'] or best_model_name in ['CNN_Syntactic_Model', 'Dense_Semantic_Model']

            # Handle dense conversion if needed for training data
            current_X_train = X_train_final_scaled
            if is_sparse_final and model_requires_dense:
                try:
                    current_X_train = current_X_train.toarray()
                except MemoryError: raise MemoryError("Cannot convert training data to dense")

            # Instantiate
            if best_model_name == 'LogisticRegression':
                final_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000)
            elif best_model_name == 'LinearSVC':
                final_model = LinearSVC(class_weight='balanced', dual='auto', random_state=RANDOM_STATE, max_iter=5000, C=1.0, tol=1e-3)
            elif best_model_name == 'RandomForest':
                final_model = RandomForestClassifier(n_estimators=150, max_depth=None, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1, min_samples_split=5)
            elif best_model_name == 'ComplementNB':
                final_model = ComplementNB()
            elif best_model_name == 'XGBoost':
                count_neg = np.sum(y_train_full == 0)
                count_pos = np.sum(y_train_full == 1)
                scale_pos_weight_val = count_neg / count_pos if count_pos > 0 else 1
                final_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, tree_method='hist', scale_pos_weight=scale_pos_weight_val, random_state=RANDOM_STATE, n_estimators=100)
            elif best_model_name == 'SVC_RBF':
                final_model = SVC(kernel='rbf', probability=False, class_weight='balanced', random_state=RANDOM_STATE) # Probability False for final model unless needed
            elif best_model_name == 'CNN_Syntactic_Model':
                # tf.random.set_seed(RANDOM_STATE)
                torch.manual_seed(RANDOM_STATE)
                final_model = CNN1D(input_dim=input_dim_final).to(DEVICE)
                is_keras_final = True
            elif best_model_name == 'Dense_Semantic_Model':
                # tf.random.set_seed(RANDOM_STATE)
                torch.manual_seed(RANDOM_STATE)
                final_model = DenseNetwork(input_dim=input_dim_final).to(DEVICE)
                is_keras_final = True
            else:
                raise ValueError(f"Unknown best model name: {best_model_name}")

            # Train
            if is_keras_final:
                # Consider using a small validation split from training data for final Keras training if desired
                # Or adjust epochs based on CV results
                final_model.fit(current_X_train, y_train_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0) # Reduced verbosity
            else:
                final_model.fit(current_X_train, y_train_full)
            print("  Training complete.")

            # --- Evaluate on Test Set ---
            print(f"  Evaluating final model on Test Set...")
            # Handle dense conversion for test data if needed
            current_X_test = test_features
            if is_sparse_final and model_requires_dense:
                try:
                    current_X_test = current_X_test.toarray()
                except MemoryError: raise MemoryError("Cannot convert test data to dense")

            acc_test, pre_test, rec_test, f1_test, _, y_pred_test = evaluate_model(
                final_model, current_X_test, y_test, is_keras=is_keras_final
            )
            print(f"  Test Set Metrics: Acc={acc_test:.4f}, P={pre_test:.4f}, R={rec_test:.4f}, F1={f1_test:.4f}")

            # --- Save Final Model ---
            model_path = save_final_model(final_model, group_name, best_model_name, final_model_dir)
            if model_path:
                saved_final_models[f"{group_name}_{best_model_name}"] = model_path

        except Exception as e:
            print(f"  ERROR during final training/evaluation for group '{group_name}', model '{best_model_name}': {e}")
            if is_keras_final and 'final_model' in locals():
                # Clear session if Keras model was involved
                from keras import backend as K
                K.clear_session()

        finally:
            # Store test results
            all_test_results.append({
                'feature_group': group_name,
                'model_name': best_model_name,
                'test_accuracy': acc_test,
                'test_precision': pre_test,
                'test_recall': rec_test,
                'test_f1_score': f1_test
            })
            # Clean up memory
            if 'final_model' in locals(): del final_model
            if 'test_features' in locals(): del test_features
            if 'current_X_train' in locals(): del current_X_train
            if 'current_X_test' in locals(): del current_X_test
            gc.collect()


    # 6. Save Test Set Results
    if all_test_results:
        test_results_df = pd.DataFrame(all_test_results)
        print(f"\nSaving test set results for best models per group to: {test_output_file}")
        test_results_df.to_csv(test_output_file, index=False)

        # 7. Identify and Report Overall Best Model based on Test Set F1
        test_results_df = test_results_df.sort_values(by='test_f1_score', ascending=False).reset_index(drop=True)
        print("\n--- Final Test Set Performance Summary (Sorted by F1-Score) ---")
        print(test_results_df)

        if not test_results_df.empty and not pd.isna(test_results_df.iloc[0]['test_f1_score']):
            best_group = test_results_df.iloc[0]['feature_group']
            best_model = test_results_df.iloc[0]['model_name']
            best_test_f1 = test_results_df.iloc[0]['test_f1_score']
            print(f"\nOverall Best Combination based on Test F1-Score: Group='{best_group}', Model='{best_model}' (F1 = {best_test_f1:.4f})")
            model_id = f"{best_group}_{best_model}"
            if model_id in saved_final_models:
                print(f"Model saved to: {saved_final_models[model_id]}")
            else:
                # Attempt to reconstruct filename if lookup fails (less reliable)
                safe_group_name = "".join(c if c.isalnum() else "_" for c in best_group)
                safe_model_name = "".join(c if c.isalnum() else "_" for c in best_model)
                print(f"Model likely saved in {final_model_dir} as final_{safe_group_name}_{safe_model_name}.[pkl/h5]")

        else:
            print("\nCould not determine the best overall model based on test results.")

    else:
        print("\nNo test results were generated.")


    print("\nPhase 3 Processing complete.")