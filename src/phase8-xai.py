import ast
import gc
import os
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
# import lime.lime_text # Keep if text-based LIME is needed later
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler, LabelEncoder # Added
from sklearn.svm import SVC, LinearSVC
from tensorflow.keras.models import load_model
from tf_keras import Sequential
from tqdm import tqdm
from xgboost import XGBClassifier

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TensorFlow verbosity
TF_AVAILABLE = True

# Configure SHAP for JS plots in notebooks (optional, can be removed if not using notebooks)
# shap.initjs() # Comment out if not running in a notebook environment

# --- Constants ---
RANDOM_STATE = 42
SEMANTIC_DIM = 384 # Default, will be updated in load_data if needed
N_SHAP_SAMPLES = 100 # Number of samples for SHAP summary plot
N_SHAP_BACKGROUND = 50 # Size of background dataset for SHAP Kernel/Deep explainer
N_LIME_SAMPLES = 5   # Number of instances to explain with LIME
N_LIME_FEATURES = 10 # Number of features to show in LIME explanations
N_IG_SAMPLES = 50    # Number of samples for Integrated Gradients

# --- Utility Functions ---

def safe_literal_eval(x):
    """Safely evaluates a string literal, returning an empty array on error."""
    try:
        if isinstance(x, str):
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.float32)
        # Handle cases where it might already be an array (e.g., if loaded differently)
        if isinstance(x, (np.ndarray, list, tuple)):
            return np.array(x, dtype=np.float32)
        return np.array([], dtype=np.float32) # Default empty array
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)

# --- Data Loading (Simplified for XAI context) ---
# Assuming data is already processed and split appropriately by previous phases.
# We just need to load the relevant data parts.

def load_data_for_xai(data_path: Path, tfidf_sparse_path: Path = None):
    """Loads processed data and optionally the sparse TF-IDF matrix."""
    print(f"Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Basic checks
    if 'label' not in df.columns:
        print("Error: 'label' column missing from data.")
        sys.exit(1)
    if 'clean_text' not in df.columns:
        print("Warning: 'clean_text' column missing. Text-based LIME might not work.")

    # Encode labels if they are not numeric
    if df['label'].dtype == 'object':
        print("Encoding labels...")
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])
        print(f"Labels mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    elif 'label_encoded' not in df.columns:
        df['label_encoded'] = df['label'] # Assume label is already encoded if numeric

    # Parse semantic vectors
    global SEMANTIC_DIM
    if 'semantic_vector' in df.columns:
        print("Parsing semantic vectors...")
        df['semantic_vector'] = df['semantic_vector'].apply(safe_literal_eval)
        # Drop rows with invalid vectors if any were created
        empty_vectors = df['semantic_vector'].apply(lambda x: x.size == 0)
        if empty_vectors.any():
            print(f"Warning: Found {empty_vectors.sum()} rows with invalid semantic vectors. Check preprocessing.")
            # Depending on XAI needs, you might drop or impute these. For simplicity, we keep them for now but they might cause errors later.
        # Update SEMANTIC_DIM based on actual data
        valid_vectors = df.loc[~empty_vectors, 'semantic_vector']
        if not valid_vectors.empty:
            first_dim = valid_vectors.iloc[0].shape[0]
            if first_dim != SEMANTIC_DIM:
                print(f"Updating SEMANTIC_DIM from {SEMANTIC_DIM} to {first_dim}")
                SEMANTIC_DIM = first_dim
            # Optional: Add check for consistent dimensions
    else:
        print("Warning: 'semantic_vector' column not found. Semantic models cannot be analyzed.")

    # Load sparse TF-IDF
    tfidf_sparse_matrix = None
    if tfidf_sparse_path and tfidf_sparse_path.exists():
        try:
            tfidf_sparse_matrix = load_npz(tfidf_sparse_path)
            print(f"Loaded sparse TF-IDF matrix with shape: {tfidf_sparse_matrix.shape}")
            if tfidf_sparse_matrix.shape[0] != len(df):
                print(f"Error: Row count mismatch between data ({len(df)}) and sparse TF-IDF ({tfidf_sparse_matrix.shape[0]}). Ensure they correspond.")
                # Consider exiting or attempting alignment if indices match
                # sys.exit(1) # More robust approach
                print("Proceeding with mismatched TF-IDF matrix. This may lead to errors.")
        except Exception as e:
            print(f"Error loading sparse TF-IDF matrix: {e}")
            tfidf_sparse_matrix = None
    elif tfidf_sparse_path:
        print(f"Warning: Sparse TF-IDF matrix file not found at {tfidf_sparse_path}.")

    # Define feature groups (adjust based on actual columns in phase2_output.csv)
    feature_groups = {
        'lexical': [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df.columns],
        'syntactic': [col for col in df.columns if col.startswith('pos_')],
        'semantic': ['semantic_vector'] if 'semantic_vector' in df.columns else [],
        'sentiment': [col for col in df.columns if col.startswith('sentiment_')],
        'tfidf_dense': [col for col in df.columns if col.startswith('tfidf_') and col not in ['tfidf', 'tfidf_feature']], # Exclude potential list columns
        'tfidf_sparse': ['tfidf_matrix.npz'] if tfidf_sparse_matrix is not None else [],
    }
    # Filter out empty groups
    feature_groups = {name: cols for name, cols in feature_groups.items() if cols}

    print("\nFeature groups identified:")
    for name, cols in feature_groups.items():
        if name == 'tfidf_sparse': print(f"- {name}: Loaded sparse matrix")
        elif name == 'semantic': print(f"- {name}: {SEMANTIC_DIM} features")
        elif name == 'tfidf_dense': print(f"- {name}: {len(cols)} dense features")
        else: print(f"- {name}: {len(cols)} features")

    # Load TF-IDF feature names if available (assuming saved from phase 2/3)
    tfidf_feature_names = None
    tfidf_names_path = data_path.parent / 'tfidf_feature_names.pkl' # Adjust path if needed
    if tfidf_names_path.exists():
        try:
            tfidf_feature_names = joblib.load(tfidf_names_path)
            print(f"Loaded {len(tfidf_feature_names)} TF-IDF feature names.")
        except Exception as e:
            print(f"Warning: Could not load TF-IDF feature names from {tfidf_names_path}: {e}")


    return df, tfidf_sparse_matrix, feature_groups, tfidf_feature_names


# --- Prediction Function Wrappers ---

def sklearn_predict_proba(model, data):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(data)
    elif hasattr(model, 'decision_function'):
        # Calibrate or use decision function directly (less ideal for probabilities)
        print("Warning: Model lacks predict_proba, using decision_function. LIME/SHAP results might be less interpretable.")
        scores = model.decision_function(data)
        if len(scores.shape) == 1: # Binary case
            probs = 1 / (1 + np.exp(-scores)) # Sigmoid
            return np.vstack([1 - probs, probs]).T
        else: # Multiclass (unlikely here but for completeness)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        # Fallback for models like ComplementNB that might only have predict
        preds = model.predict(data)
        # Convert predictions to pseudo-probabilities (0 or 1)
        return np.eye(2)[preds] # Assumes binary classification [class0, class1]

def keras_predict_proba(model, data):
    if isinstance(data, list): # Handle multi-input models
        return model.predict(data).flatten()
    else:
        return model.predict(data).flatten()

# def pytorch_predict_proba(model, data):
#     if not PYTORCH_AVAILABLE:
#         raise ImportError("PyTorch is not available.")
#     model.eval()
#     device = next(model.parameters()).device # Get device model is on
#     with torch.no_grad():
#         if isinstance(data, list): # Handle multi-input models (needs specific adaptation)
#             # This part needs to be customized based on how the PyTorch fusion model expects input
#             # Example: Assuming first is semantic, second is syntactic
#             tensor_inputs = [torch.tensor(d, dtype=torch.float32).to(device) for d in data]
#             probs = model(*tensor_inputs)
#         else:
#             inputs = torch.tensor(data, dtype=torch.float32).to(device)
#             probs = model(inputs)
#         # Ensure output is probabilities (apply sigmoid if model output are logits)
#         if not ((probs >= 0) & (probs <= 1)).all():
#             probs = torch.sigmoid(probs)
#         return probs.cpu().numpy().flatten() # Return as numpy array


class XAIAnalyzer:
    def __init__(self, model_paths, scaler_paths, data_path, tfidf_sparse_path=None, output_dir='results/phase8'):
        self.data_path = data_path
        self.tfidf_sparse_path = tfidf_sparse_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df, self.tfidf_sparse_matrix, self.feature_groups, self.tfidf_feature_names = load_data_for_xai(data_path, tfidf_sparse_path)
        self.models = self._load_models(model_paths)
        self.scalers = self._load_scalers(scaler_paths)
        self.explanations = defaultdict(dict) # Use defaultdict

        # Store feature names for different groups
        self.feature_names = {}
        self.feature_names['lexical'] = self.feature_groups.get('lexical', [])
        self.feature_names['syntactic'] = self.feature_groups.get('syntactic', [])
        self.feature_names['sentiment'] = self.feature_groups.get('sentiment', [])
        self.feature_names['tfidf_dense'] = self.feature_groups.get('tfidf_dense', [])
        self.feature_names['semantic'] = [f'sem_{i}' for i in range(SEMANTIC_DIM)] # Generic names for semantic
        self.feature_names['tfidf_sparse'] = self.tfidf_feature_names if self.tfidf_feature_names is not None else \
            [f'tfidf_{i}' for i in range(self.tfidf_sparse_matrix.shape[1])] if self.tfidf_sparse_matrix is not None else []

        # Combine dense features for easier access
        self.dense_feature_names = self.feature_names['lexical'] + \
                                   self.feature_names['syntactic'] + \
                                   self.feature_names['sentiment'] + \
                                   self.feature_names['tfidf_dense']


    def _load_models(self, model_paths):
        """Load trained models from different phases"""
        models = {}
        print("\nLoading models...")
        for name, path in model_paths.items():
            path = Path(path) # Ensure path is a Path object
            try:
                if path.suffix == '.pkl':
                    models[name] = joblib.load(path)
                    print(f"  Loaded sklearn/joblib model '{name}' from {path}")
                elif path.suffix == '.h5' and TF_AVAILABLE:
                    models[name] = load_model(path)
                    print(f"  Loaded Keras model '{name}' from {path}")
                # Add PyTorch loading if needed - assuming joblib dump based on phase3 example
                # elif path.suffix == '.pt' or path.suffix == '.pth':
                #     if PYTORCH_AVAILABLE:
                #         # This requires knowing the model class definition
                #         # Option 1: If saved entire model object (less common, potentially fragile)
                #         # models[name] = torch.load(path)
                #         # Option 2: If saved state_dict (preferred) - REQUIRES model class definition
                #         # model_instance = YourModelClass(...) # Instantiate model first
                #         # model_instance.load_state_dict(torch.load(path))
                #         # models[name] = model_instance
                #         print(f"  PyTorch model loading for {name} needs specific class definition - Skipping for now.")
                #         # If phase3 truly used joblib for PyTorch:
                #         # models[name] = joblib.load(path)
                        # print(f"  Loaded PyTorch (via joblib) model '{name}' from {path}")
                    # else:
                    #     print(f"  Skipping PyTorch model '{name}' due to missing library.")

            except FileNotFoundError:
                print(f"  Error: Model file not found for '{name}' at {path}")
            except Exception as e:
                print(f"  Error loading model '{name}' from {path}: {e}")
        if not models:
            print("Error: No models were loaded successfully. Exiting.")
            sys.exit(1)
        return models

    def _load_scalers(self, scaler_paths):
        """Load saved scalers"""
        scalers = {}
        print("\nLoading scalers...")
        for name, path in scaler_paths.items():
            path = Path(path)
            if path.exists():
                try:
                    scalers[name] = joblib.load(path)
                    print(f"  Loaded scaler '{name}' from {path}")
                except Exception as e:
                    print(f"  Error loading scaler '{name}' from {path}: {e}")
            else:
                print(f"  Warning: Scaler file not found for '{name}' at {path}")
        return scalers


    def _get_predict_fn(self, model_name):
        """Get the appropriate prediction function for the model"""
        if model_name not in self.models:
            print(f"Warning: Model '{model_name}' not loaded.")
            return None

        model = self.models[model_name]
        model_module = type(model).__module__

        # Check for scikit-learn models (including those saved via joblib)
        if model_module.startswith('sklearn') or isinstance(model, (joblib.parallel.Parallel, SVC, LinearSVC, LogisticRegression, RandomForestClassifier, ComplementNB)): # Add other sklearn types if needed
            # Special case for ComplementNB which might be wrapped if sparse
            if hasattr(model, 'steps'): # Pipeline
                final_estimator = model.steps[-1][1]
                if isinstance(final_estimator, ComplementNB):
                    # Need to handle sparse data transformation within the wrapper
                    def predict_proba_wrapper(data):
                        # Assuming data is already sparse if required by pipeline
                        return sklearn_predict_proba(model, data)
                    return predict_proba_wrapper
                else:
                    return lambda data: sklearn_predict_proba(model, data)
            else: # Simple sklearn model
                return lambda data: sklearn_predict_proba(model, data)
        # Check for Keras models (TensorFlow backend)
        elif TF_AVAILABLE and ('tensorflow.python.keras' in model_module or 'keras.src.models' in model_module or isinstance(model, tf.keras.Model)):
            return lambda data: keras_predict_proba(model, data)
        # Check for PyTorch models (assuming loaded via joblib or torch.load)
        # elif PYTORCH_AVAILABLE and ('torch.nn.modules' in model_module or isinstance(model, torch.nn.Module)):
        #     return lambda data: pytorch_predict_proba(model, data)
        else:
            print(f"Warning: Unsupported model type ('{model_module}') or missing library for {model_name}. Cannot create prediction function.")
            return None
    # def _get_predict_fn(self, model_name):
    #     """Get the appropriate prediction function for the model"""
    #     model = self.models[model_name]
    #     model_type = type(model).__module__.split('.')[0] # e.g., 'sklearn', 'tensorflow', 'torch'
    #
    #     if model_type == 'sklearn' or isinstance(model, joblib.parallel.Parallel): # Handle sklearn pipelines/models
    #         # Special case for ComplementNB which might be wrapped if sparse
    #         if hasattr(model, 'steps'): # Pipeline
    #             final_estimator = model.steps[-1][1]
    #             if isinstance(final_estimator, ComplementNB):
    #                 # Need to handle sparse data transformation within the wrapper
    #                 def predict_proba_wrapper(data):
    #                     # Assuming data is already sparse if required by pipeline
    #                     return sklearn_predict_proba(model, data)
    #                 return predict_proba_wrapper
    #             else:
    #                 return lambda data: sklearn_predict_proba(model, data)
    #         else: # Simple sklearn model
    #             return lambda data: sklearn_predict_proba(model, data)
    #     elif model_type == 'tensorflow' and TF_AVAILABLE:
    #         return lambda data: keras_predict_proba(model, data)
    #     # elif model_type == 'torch' and PYTORCH_AVAILABLE:
    #     #     return lambda data: pytorch_predict_proba(model, data)
    #     else:
    #         print(f"Warning: Unsupported model type or missing library for {model_name}. Cannot create prediction function.")
    #         return None

    def _prepare_data_for_model(self, feature_group_keys, sample_indices=None, scale=True, return_dense_array=False):
        """
        Prepares data (dense, sparse, or combined) for a specific model based on feature group keys.

        Args:
            feature_group_keys (list): List of keys from self.feature_groups (e.g., ['lexical', 'syntactic']).
            sample_indices (np.array, optional): Indices to sample from the data. Defaults to None (use all data).
            scale (bool): Whether to apply scaling to dense features. Defaults to True.
            return_dense_array (bool): For sparse features, whether to convert to dense. Defaults to False.

        Returns:
            tuple: (prepared_data, feature_names_list)
                   prepared_data can be np.ndarray, scipy.sparse matrix, or list (for multi-input models).
                   feature_names_list contains the names corresponding to the columns/dimensions.
        """
        df_subset = self.df if sample_indices is None else self.df.iloc[sample_indices].reset_index(drop=True)
        tfidf_sparse_subset = self.tfidf_sparse_matrix if sample_indices is None else self.tfidf_sparse_matrix[sample_indices]

        prepared_parts = []
        feature_names_list = []
        requires_sparse = False

        for key in feature_group_keys:
            if key == 'tfidf_sparse':
                if tfidf_sparse_subset is None:
                    print(f"Warning: Requested tfidf_sparse but matrix not loaded.")
                    continue
                prepared_parts.append(tfidf_sparse_subset)
                feature_names_list.extend(self.feature_names['tfidf_sparse'])
                requires_sparse = True
            elif key == 'semantic':
                if 'semantic' not in self.feature_groups:
                    print(f"Warning: Requested semantic features but not found in data.")
                    continue
                data_raw = np.stack(df_subset['semantic_vector'].values)
                scaler = self.scalers.get('semantic') # Use 'semantic' or specific key if saved differently
                if scaler and scale:
                    data_scaled = scaler.transform(data_raw)
                    prepared_parts.append(data_scaled)
                else:
                    if scale: print(f"Warning: Scaler for '{key}' not found. Using unscaled data.")
                    prepared_parts.append(data_raw)
                feature_names_list.extend(self.feature_names['semantic'])
            else: # Dense features (lexical, syntactic, sentiment, tfidf_dense)
                cols = self.feature_groups.get(key)
                if not cols:
                    print(f"Warning: Feature group '{key}' not found or empty.")
                    continue
                data_raw = df_subset[cols].values.astype(np.float32)
                # Handle potential NaNs introduced if safe_literal_eval failed earlier
                if np.isnan(data_raw).any():
                    print(f"Warning: NaNs found in group '{key}'. Imputing with 0.")
                    data_raw = np.nan_to_num(data_raw)

                scaler = self.scalers.get(key)
                if scaler and scale:
                    data_scaled = scaler.transform(data_raw)
                    prepared_parts.append(data_scaled)
                else:
                    if scale: print(f"Warning: Scaler for '{key}' not found. Using unscaled data.")
                    prepared_parts.append(data_raw)
                feature_names_list.extend(self.feature_names[key])

        if not prepared_parts:
            return None, []

        # Combine features
        if len(prepared_parts) == 1:
            final_data = prepared_parts[0]
            if requires_sparse and return_dense_array:
                try:
                    final_data = final_data.toarray()
                except MemoryError:
                    print("MemoryError: Cannot convert sparse matrix to dense array.")
                    return None, [] # Or handle differently
        else:
            # Check if sparse matrix is involved
            sparse_indices = [i for i, part in enumerate(prepared_parts) if hasattr(part, 'tocsr')]
            if len(sparse_indices) > 1:
                print("Error: Cannot directly combine multiple sparse matrices with dense ones using hstack. Revise strategy.")
                return None, []
            elif len(sparse_indices) == 1:
                sparse_idx = sparse_indices[0]
                sparse_mat = prepared_parts.pop(sparse_idx)
                dense_mat = np.hstack(prepared_parts) if prepared_parts else None

                if dense_mat is not None:
                    # Combine sparse and dense
                    from scipy.sparse import hstack as sparse_hstack
                    final_data = sparse_hstack([sparse_mat, dense_mat], format='csr')
                    # Adjust feature names order if sparse wasn't first
                    if sparse_idx != 0:
                        sparse_names = self.feature_names['tfidf_sparse']
                        dense_names = [name for i, part_names in enumerate(feature_names_list) if i != sparse_idx for name in part_names]
                        feature_names_list = sparse_names + dense_names # Assuming sparse comes first in hstack
                else:
                    final_data = sparse_mat # Only sparse was present

                if return_dense_array:
                    try:
                        final_data = final_data.toarray()
                    except MemoryError:
                        print("MemoryError: Cannot convert combined sparse matrix to dense array.")
                        return None, []
            else: # All parts are dense
                final_data = np.hstack(prepared_parts)

        return final_data, feature_names_list


    def shap_analysis(self, model_name, feature_group_keys, sample_size=N_SHAP_SAMPLES, background_size=N_SHAP_BACKGROUND):
        """Perform SHAP analysis on selected model using specified features."""
        print(f"\n--- Performing SHAP analysis for {model_name} ({', '.join(feature_group_keys)}) ---")
        if model_name not in self.models:
            print(f"Model '{model_name}' not loaded. Skipping SHAP.")
            return

        model = self.models[model_name]
        predict_fn = self._get_predict_fn(model_name)
        if predict_fn is None:
            print("Cannot get prediction function. Skipping SHAP.")
            return

        # Determine if the model requires dense input (even if underlying features are sparse)
        model_type = type(model).__module__.split('.')[0]
        # Keras/PyTorch models typically need dense, some sklearn models too
        requires_dense_input = model_type in ['tensorflow', 'torch'] or \
                               isinstance(model, (joblib.parallel.Parallel, RandomForestClassifier, XGBClassifier)) # Add others if needed

        # Prepare data
        data, feature_names = self._prepare_data_for_model(
            feature_group_keys,
            scale=True, # SHAP generally works better with scaled data
            return_dense_array=requires_dense_input # Convert to dense if model needs it
        )
        if data is None:
            print("Data preparation failed. Skipping SHAP.")
            return

        is_sparse_data = hasattr(data, 'tocsr') and not requires_dense_input

        # Sample data for explanation
        if data.shape[0] <= sample_size:
            sample_indices = np.arange(data.shape[0])
        else:
            sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)

        data_sample = data[sample_indices]

        # Prepare background data
        if data.shape[0] <= background_size:
            background_indices = np.arange(data.shape[0])
        else:
            # Ensure background doesn't overlap significantly with sample for KernelExplainer
            available_indices = np.setdiff1d(np.arange(data.shape[0]), sample_indices)
            bg_choice_size = min(background_size, len(available_indices))
            if bg_choice_size < background_size: print(f"Warning: Using smaller background ({bg_choice_size}) due to sample size.")
            background_indices = np.random.choice(available_indices, bg_choice_size, replace=False) if bg_choice_size > 0 else sample_indices[:min(background_size, len(sample_indices))] # Fallback if no non-overlapping data

        background_data = data[background_indices]

        # Choose SHAP explainer
        explainer = None
        shap_values = None
        try:
            if model_type == 'tensorflow' and TF_AVAILABLE and not isinstance(model, Sequential): # DeepExplainer works best for non-Sequential TF
                # DeepExplainer expects tensors
                background_tensor = tf.convert_to_tensor(background_data)
                data_sample_tensor = tf.convert_to_tensor(data_sample)
                explainer = shap.DeepExplainer(model, background_tensor)
                shap_values = explainer.shap_values(data_sample_tensor)
                # Often returns list [neg_class_shap, pos_class_shap], take positive class
                if isinstance(shap_values, list): shap_values = shap_values[1]

            # Fallback to KernelExplainer for sklearn, PyTorch, Sequential TF, or if DeepExplainer fails
            # KernelExplainer needs a function that takes a numpy array and returns probabilities
            if explainer is None:
                # Wrap predict_fn for KernelExplainer (needs 2D output: [prob_class_0, prob_class_1])
                def wrapped_predict_proba(data_input):
                    probs_flat = predict_fn(data_input)
                    return np.vstack([1 - probs_flat, probs_flat]).T

                # Use kmeans for background summary if data is large
                if background_data.shape[0] > N_SHAP_BACKGROUND:
                    try:
                        background_summary = shap.kmeans(background_data, N_SHAP_BACKGROUND)
                        print(f"  Summarized background data using kmeans ({N_SHAP_BACKGROUND} centers).")
                    except Exception as kmeans_err:
                        print(f"  KMeans background summarization failed: {kmeans_err}. Using random sample.")
                        background_summary = background_data # Use the originally sampled background
                else:
                    background_summary = background_data

                print(f"  Using KernelExplainer with background shape: {background_summary.shape}")
                explainer = shap.KernelExplainer(wrapped_predict_proba, background_summary)

                print(f"  Calculating SHAP values for sample shape: {data_sample.shape}")
                # Link function depends on output (logit for TF/PyTorch often, identity for sklearn proba)
                link = "logit" if model_type in ['tensorflow', 'torch'] else "identity"
                shap_values = explainer.shap_values(data_sample, l1_reg="auto", nsamples='auto') # nsamples='auto' is often good

                # KernelExplainer returns list [neg_class_shap, pos_class_shap] for binary
                if isinstance(shap_values, list): shap_values = shap_values[1]

            if shap_values is None:
                print("SHAP values could not be computed.")
                return

            print("  SHAP values computed.")
            self.explanations[model_name]['shap'] = shap_values
            self.explanations[model_name]['shap_feature_names'] = feature_names
            self.explanations[model_name]['shap_data_sample'] = data_sample # Store data used

            # Visualization
            plt.figure()
            shap.summary_plot(shap_values, data_sample, feature_names=feature_names, max_display=20, show=False)
            plt.title(f'SHAP Summary Plot for {model_name}')
            plt.tight_layout()
            save_path = self.output_dir / f'shap_summary_{model_name}.png'
            plt.savefig(save_path)
            print(f"  Saved SHAP summary plot to {save_path}")
            plt.close()

            # Example force plot for the first instance
            try:
                # Create Explanation object for better plotting
                expl = shap.Explanation(values=shap_values[0],
                                        base_values=explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value)>1 else explainer.expected_value, # Use positive class base value if available
                                        data=data_sample[0],
                                        feature_names=feature_names)
                plt.figure()
                # Use matplotlib=True for saving non-interactive plot
                shap.force_plot(expl.base_values, expl.values, expl.data, feature_names=feature_names, matplotlib=True, show=False)
                plt.title(f'SHAP Force Plot (Instance 0) for {model_name}')
                plt.tight_layout()
                save_path_force = self.output_dir / f'shap_force_instance0_{model_name}.png'
                plt.savefig(save_path_force)
                print(f"  Saved SHAP force plot example to {save_path_force}")
                plt.close()
            except Exception as force_plot_err:
                print(f"  Could not generate SHAP force plot: {force_plot_err}")


        except Exception as e:
            print(f"  ERROR during SHAP analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
        finally:
            del data, data_sample, background_data, explainer, shap_values
            gc.collect()


    def lime_analysis(self, model_name, feature_group_keys, num_samples=N_LIME_SAMPLES, num_features=N_LIME_FEATURES):
        """Perform LIME analysis on selected model using specified features."""
        print(f"\n--- Performing LIME analysis for {model_name} ({', '.join(feature_group_keys)}) ---")
        if model_name not in self.models:
            print(f"Model '{model_name}' not loaded. Skipping LIME.")
            return

        model = self.models[model_name]
        predict_fn = self._get_predict_fn(model_name)
        if predict_fn is None:
            print("Cannot get prediction function. Skipping LIME.")
            return

        # LIME Tabular works with dense data.
        data, feature_names = self._prepare_data_for_model(
            feature_group_keys,
            scale=True, # LIME often benefits from scaling
            return_dense_array=True # LIME Tabular requires dense numpy array
        )
        if data is None:
            print("Data preparation failed. Skipping LIME.")
            return

        # Check if data preparation was successful and feature names match
        if data.shape[1] != len(feature_names):
            print(f"Error: Mismatch between data columns ({data.shape[1]}) and feature names ({len(feature_names)}). Skipping LIME.")
            return

        # Sample instances to explain
        if data.shape[0] <= num_samples:
            sample_indices = np.arange(data.shape[0])
        else:
            sample_indices = np.random.choice(data.shape[0], num_samples, replace=False)

        data_sample = data[sample_indices]

        # Create LIME Explainer
        try:
            # Wrap predict_fn for LIME (needs 2D output: [prob_class_0, prob_class_1])
            def wrapped_predict_proba(data_input):
                # LIME might pass data in a slightly different format, ensure it's 2D numpy
                if not isinstance(data_input, np.ndarray):
                    data_input = np.array(data_input)
                if data_input.ndim == 1:
                    data_input = data_input.reshape(1, -1)
                probs_flat = predict_fn(data_input)
                # Ensure probs_flat has the correct shape before vstack
                if probs_flat.ndim == 0: # Handle single prediction case
                    probs_flat = np.array([probs_flat])
                elif probs_flat.ndim > 1:
                    probs_flat = probs_flat.flatten() # Ensure it's 1D

                return np.vstack([1 - probs_flat, probs_flat]).T

            print(f"  Initializing LimeTabularExplainer with training data shape: {data.shape}")
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data, # Use full (or large sample) data for learning distributions
                feature_names=feature_names,
                class_names=['non-harmful', 'harmful'], # Adjust if needed
                mode='classification',
                random_state=RANDOM_STATE
            )

            print(f"  Explaining {num_samples} instances...")
            lime_explanations = []
            for i in tqdm(range(num_samples)):
                instance = data_sample[i]
                try:
                    exp = explainer.explain_instance(
                        instance,
                        wrapped_predict_proba,
                        num_features=num_features,
                        top_labels=1 # Explain the predicted class
                    )
                    lime_explanations.append(exp)

                    # Save example visualization
                    if i < 3: # Save first few examples
                        try:
                            fig = exp.as_pyplot_figure()
                            fig.suptitle(f'LIME Explanation (Instance {sample_indices[i]}) for {model_name}')
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
                            save_path = self.output_dir / f'lime_{model_name}_instance{sample_indices[i]}.png'
                            fig.savefig(save_path)
                            print(f"    Saved LIME plot for instance {sample_indices[i]} to {save_path}")
                            plt.close(fig)
                        except Exception as plot_err:
                            print(f"    Warning: Could not save LIME plot for instance {sample_indices[i]}: {plot_err}")
                            plt.close('all') # Close any potentially open figures

                except Exception as explain_err:
                    print(f"    Error explaining instance {sample_indices[i]}: {explain_err}")

            self.explanations[model_name]['lime'] = lime_explanations
            print("  LIME analysis complete.")

        except Exception as e:
            print(f"  ERROR during LIME analysis setup for {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del data, data_sample, explainer
            gc.collect()


    def integrated_gradients_analysis(self, model_name, feature_group_keys, num_samples=N_IG_SAMPLES):
        """Perform Integrated Gradients analysis for TensorFlow models."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping Integrated Gradients.")
            return
        if model_name not in self.models or not isinstance(self.models[model_name], tf.keras.Model):
            print(f"Model '{model_name}' is not a TensorFlow/Keras model or not loaded. Skipping IG.")
            return

        print(f"\n--- Performing Integrated Gradients for {model_name} ({', '.join(feature_group_keys)}) ---")
        model = self.models[model_name]

        # Prepare data - IG needs dense data
        data, feature_names = self._prepare_data_for_model(
            feature_group_keys,
            scale=True,
            return_dense_array=True
        )
        if data is None:
            print("Data preparation failed. Skipping IG.")
            return

        # Check for data/feature name mismatch
        if data.shape[1] != len(feature_names):
            print(f"Error: Mismatch between data columns ({data.shape[1]}) and feature names ({len(feature_names)}). Skipping IG.")
            return

        # Sample data
        if data.shape[0] <= num_samples:
            sample_indices = np.arange(data.shape[0])
        else:
            sample_indices = np.random.choice(data.shape[0], num_samples, replace=False)

        data_sample = data[sample_indices]

        try:
            inputs = tf.convert_to_tensor(data_sample, dtype=tf.float32)
            # Baseline: Use zeros or mean (zeros is common)
            baseline = tf.zeros_like(inputs)
            # baseline = tf.convert_to_tensor(np.mean(data, axis=0, keepdims=True), dtype=tf.float32) # Alternative: mean baseline

            m_steps = 50 # Number of steps for approximation
            alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate alphas

            interpolated_inputs = []
            for alpha in alphas:
                interpolated_inputs.append(baseline + alpha * (inputs - baseline))
            interpolated_inputs = tf.stack(interpolated_inputs) # Shape: (m_steps+1, num_samples, num_features)

            all_grads = []
            # Process in batches if memory is a concern
            batch_size = 32
            num_batches = int(np.ceil(interpolated_inputs.shape[0] / batch_size))

            print(f"  Calculating gradients for {m_steps+1} interpolated inputs...")
            for i in tqdm(range(num_batches)):
                batch_inputs = interpolated_inputs[i*batch_size:(i+1)*batch_size]
                with tf.GradientTape() as tape:
                    tape.watch(batch_inputs)
                    # Handle potential multi-input models if necessary here
                    batch_preds = model(batch_inputs, training=False) # Shape: (batch_size, num_samples, 1) or similar
                    # We need gradients w.r.t input features for each sample's prediction
                    # Sum predictions if needed, or select the target class output
                    target_preds = tf.reduce_sum(batch_preds, axis=-1) # Sum over output dimension if needed

                # Gradients shape: (batch_size, num_samples, num_features)
                batch_grads = tape.gradient(target_preds, batch_inputs)
                if batch_grads is None:
                    print(f"Error: Gradients are None for batch {i}. Check model connectivity or input.")
                    return # Stop if gradients fail
                all_grads.append(batch_grads.numpy())

            if not all_grads:
                print("Error: No gradients were computed.")
                return

            grads_tensor = np.concatenate(all_grads, axis=0) # Shape: (m_steps+1, num_samples, num_features)

            # Average gradients across steps (Trapezoidal rule approximation)
            avg_grads = (grads_tensor[:-1] + grads_tensor[1:]) / 2.0
            avg_grads = np.mean(avg_grads, axis=0) # Shape: (num_samples, num_features)

            # Calculate Integrated Gradients
            integrated_gradients = (inputs.numpy() - baseline.numpy()) * avg_grads # Shape: (num_samples, num_features)

            print("  Integrated Gradients computed.")
            self.explanations[model_name]['ig'] = integrated_gradients
            self.explanations[model_name]['ig_feature_names'] = feature_names
            self.explanations[model_name]['ig_data_sample'] = data_sample

            # Aggregate and visualize
            attributions = np.mean(np.abs(integrated_gradients), axis=0) # Mean absolute attribution per feature
            top_n = min(20, len(feature_names)) # Show top 20 features or fewer
            top_indices = np.argsort(attributions)[-top_n:]
            top_attr = attributions[top_indices]
            top_names = [feature_names[i] for i in top_indices]

            plt.figure(figsize=(8, max(6, top_n // 2))) # Adjust figure size
            plt.barh(range(top_n), top_attr, align='center')
            plt.yticks(range(top_n), top_names)
            plt.xlabel('Mean Absolute Integrated Gradient')
            plt.title(f'Integrated Gradients Feature Importance for {model_name}')
            plt.tight_layout()
            save_path = self.output_dir / f'ig_summary_{model_name}.png'
            plt.savefig(save_path)
            print(f"  Saved Integrated Gradients summary plot to {save_path}")
            plt.close()

        except Exception as e:
            print(f"  ERROR during Integrated Gradients analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del data, data_sample, baseline, inputs, integrated_gradients
            if 'grads_tensor' in locals(): del grads_tensor
            if 'avg_grads' in locals(): del avg_grads
            gc.collect()
            if TF_AVAILABLE: tf.keras.backend.clear_session()


    def analyze_feature_categories(self):
        """Analyze feature category impacts across models using SHAP or IG."""
        print("\n--- Analyzing Feature Category Impacts ---")
        category_impacts = {}
        processed_models = set()

        for model_name, explanation_data in self.explanations.items():
            impact_source = None
            values = None
            f_names = None

            if 'shap' in explanation_data:
                impact_source = 'SHAP'
                # Use mean absolute SHAP value across samples
                values = np.mean(np.abs(explanation_data['shap']), axis=0)
                f_names = explanation_data.get('shap_feature_names')
            elif 'ig' in explanation_data: # Use IG if SHAP is not available
                impact_source = 'IG'
                # Use mean absolute IG value across samples
                values = np.mean(np.abs(explanation_data['ig']), axis=0)
                f_names = explanation_data.get('ig_feature_names')

            if values is not None and f_names is not None:
                print(f"  Aggregating category impacts for {model_name} using {impact_source}...")
                if len(values) != len(f_names):
                    print(f"    Warning: Mismatch between {impact_source} values ({len(values)}) and feature names ({len(f_names)}) for {model_name}. Skipping category analysis.")
                    continue
                category_impacts[model_name] = self._aggregate_by_category(values, f_names)
                processed_models.add(model_name)
            else:
                print(f"  No suitable explanation data (SHAP or IG) found for {model_name} to analyze categories.")

        if not category_impacts:
            print("No category impacts could be calculated.")
            return

        # Visualization
        try:
            df_impacts = pd.DataFrame(category_impacts).fillna(0)
            if not df_impacts.empty:
                ax = df_impacts.plot(kind='bar', figsize=(12, 7))
                plt.title('Feature Category Importance Across Models (Mean Abs SHAP/IG)')
                plt.ylabel('Mean Absolute Attribution Score')
                plt.xlabel('Feature Category')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                save_path = self.output_dir / 'category_impacts_comparison.png'
                plt.savefig(save_path)
                print(f"Saved category impacts plot to {save_path}")
                plt.close()
            else:
                print("No data to plot for category impacts.")
        except Exception as e:
            print(f"Error plotting category impacts: {e}")
            plt.close('all')


    def _aggregate_by_category(self, values, feature_names):
        """Aggregate feature importance by predefined categories using feature names."""
        impacts = defaultdict(float)
        name_to_index = {name: i for i, name in enumerate(feature_names)}

        for category, defined_features in self.feature_groups.items():
            category_total_impact = 0.0
            count = 0
            if category == 'semantic': # Handle generic semantic names
                indices = [idx for name, idx in name_to_index.items() if name.startswith('sem_')]
            elif category == 'tfidf_sparse' or category == 'tfidf_dense': # Handle generic tfidf names
                indices = [idx for name, idx in name_to_index.items() if name.startswith('tfidf_')]
                # Refine if actual tfidf names are available and match
                if self.tfidf_feature_names and category == 'tfidf_sparse':
                    indices = [idx for name, idx in name_to_index.items() if name in self.tfidf_feature_names]
                elif category == 'tfidf_dense':
                    cols_in_group = self.feature_groups.get('tfidf_dense', [])
                    indices = [idx for name, idx in name_to_index.items() if name in cols_in_group]

            else: # Lexical, Syntactic, Sentiment
                indices = [name_to_index[name] for name in defined_features if name in name_to_index]

            if indices:
                category_total_impact = np.sum(values[indices])
                count = len(indices)

            impacts[category] = category_total_impact
            # print(f"    Category '{category}': Found {count} features, Total Impact: {category_total_impact:.4f}")

        # Normalize or report raw sums? Reporting raw sums for now.
        return dict(impacts)


    def generate_report(self):
        """Generate comprehensive XAI report summarizing findings."""
        print("\n--- Generating XAI Report ---")
        report_data = []
        summary_insights = []

        for model_name in self.models.keys():
            if model_name not in self.explanations:
                print(f"  No explanation data found for {model_name}. Skipping in report.")
                continue

            model_exps = self.explanations[model_name]
            model_report = {'model': model_name}
            summary_insights.append(f"Model: {model_name}")
            summary_insights.append("-" * len(f"Model: {model_name}"))

            top_features_combined = defaultdict(lambda: {'shap': 0.0, 'lime': 0.0, 'ig': 0.0, 'count': 0})

            # Process SHAP
            if 'shap' in model_exps:
                shap_values_abs_mean = np.mean(np.abs(model_exps['shap']), axis=0)
                f_names = model_exps.get('shap_feature_names')
                if f_names and len(shap_values_abs_mean) == len(f_names):
                    indices = np.argsort(shap_values_abs_mean)[::-1] # Descending order
                    model_report['shap_top_features'] = {f_names[i]: shap_values_abs_mean[i] for i in indices[:N_LIME_FEATURES]}
                    summary_insights.append("  Top Features (SHAP - Mean Abs Value):")
                    summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in model_report['shap_top_features'].items()])
                    # Aggregate for combined view
                    for i in indices: top_features_combined[f_names[i]]['shap'] += shap_values_abs_mean[i]; top_features_combined[f_names[i]]['count'] +=1

                    # Category impacts from SHAP
                    category_impacts = self._aggregate_by_category(shap_values_abs_mean, f_names)
                    model_report['shap_category_impacts'] = category_impacts
                    summary_insights.append("  Feature Category Impacts (SHAP):")
                    summary_insights.extend([f"    - {cat}: {impact:.4f}" for cat, impact in sorted(category_impacts.items(), key=lambda item: item[1], reverse=True)])
                else:
                    summary_insights.append("  SHAP: Feature names mismatch or missing.")


            # Process LIME
            if 'lime' in model_exps:
                lime_feature_scores = self._process_lime_explanations(model_exps['lime'])
                model_report['lime_top_features'] = lime_feature_scores
                summary_insights.append(f"\n  Top Features (LIME - Aggregated Abs Score over {len(model_exps['lime'])} instances):")
                summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in lime_feature_scores.items()])
                # Aggregate for combined view
                for feat, score in lime_feature_scores.items(): top_features_combined[feat]['lime'] += score; top_features_combined[feat]['count'] +=1

            # Process Integrated Gradients
            if 'ig' in model_exps:
                ig_values_abs_mean = np.mean(np.abs(model_exps['ig']), axis=0)
                f_names = model_exps.get('ig_feature_names')
                if f_names and len(ig_values_abs_mean) == len(f_names):
                    indices = np.argsort(ig_values_abs_mean)[::-1]
                    model_report['ig_top_features'] = {f_names[i]: ig_values_abs_mean[i] for i in indices[:N_LIME_FEATURES]}
                    summary_insights.append("\n  Top Features (Integrated Gradients - Mean Abs Value):")
                    summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in model_report['ig_top_features'].items()])
                    # Aggregate for combined view
                    for i in indices: top_features_combined[f_names[i]]['ig'] += ig_values_abs_mean[i]; top_features_combined[f_names[i]]['count'] +=1

                    # Category impacts from IG (if SHAP wasn't available)
                    if 'shap_category_impacts' not in model_report:
                        category_impacts = self._aggregate_by_category(ig_values_abs_mean, f_names)
                        model_report['ig_category_impacts'] = category_impacts
                        summary_insights.append("  Feature Category Impacts (IG):")
                        summary_insights.extend([f"    - {cat}: {impact:.4f}" for cat, impact in sorted(category_impacts.items(), key=lambda item: item[1], reverse=True)])
                else:
                    summary_insights.append("  IG: Feature names mismatch or missing.")

            # Calculate combined ranking (simple average score across methods where available)
            avg_scores = {feat: (d['shap'] + d['lime'] + d['ig']) / d['count'] for feat, d in top_features_combined.items() if d['count'] > 0}
            sorted_combined = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)[:N_LIME_FEATURES]
            model_report['combined_top_features'] = dict(sorted_combined)
            summary_insights.append("\n  Top Features (Combined Average Score):")
            summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in sorted_combined])

            report_data.append(model_report)
            summary_insights.append("\n" + "="*50 + "\n")

        # Save detailed report data (might be complex structure)
        try:
            # Convert complex dicts to strings for simple CSV saving
            report_df_save = pd.DataFrame(report_data)
            for col in report_df_save.columns:
                if isinstance(report_df_save[col].iloc[0], dict):
                    report_df_save[col] = report_df_save[col].apply(str)
            report_path = self.output_dir / 'xai_report_summary.csv'
            report_df_save.to_csv(report_path, index=False)
            print(f"Saved summary XAI report to {report_path}")
        except Exception as e:
            print(f"Error saving summary CSV report: {e}")

        # Save text insights
        insights_path = self.output_dir / 'key_xai_insights.txt'
        with open(insights_path, 'w') as f:
            f.write("\n".join(summary_insights))
        print(f"Saved key insights to {insights_path}")


    def _process_lime_explanations(self, lime_exps):
        """Extract and aggregate top features from LIME explanations."""
        feature_scores = defaultdict(float)
        feature_counts = defaultdict(int)
        total_exps = len(lime_exps)

        for exp in lime_exps:
            # exp.as_list() returns tuples like ('feature_name <= value', score) or ('value < feature_name', score)
            # We need to robustly extract the feature name.
            for feature_condition, score in exp.as_list(label=1): # Assuming label=1 is the 'harmful' class
                # Basic parsing - might need adjustment based on LIME output format
                parts = feature_condition.split(' ')
                feature_name = None
                for part in parts:
                    # Check if part matches any known feature name exactly
                    if part in self.dense_feature_names or part in self.feature_names['semantic'] or part in self.feature_names['tfidf_sparse']:
                        feature_name = part
                        break
                    # Fallback: Check if a known feature name is a substring (less reliable)
                    # This is tricky, especially with TF-IDF names.
                    # A better approach might be needed if LIME uses complex conditions.
                if feature_name is None:
                    feature_name = parts[0] # Best guess if exact match fails

                feature_scores[feature_name] += abs(score)
                feature_counts[feature_name] += 1

        # Average score per feature (optional, could also use sum)
        # avg_feature_scores = {feat: feature_scores[feat] / feature_counts[feat] for feat in feature_scores}
        # Using sum for now as it reflects total contribution across explanations
        avg_feature_scores = {feat: feature_scores[feat] / total_exps for feat in feature_scores}


        # Return top N features based on aggregated score
        return dict(sorted(avg_feature_scores.items(), key=lambda item: item[1], reverse=True)[:N_LIME_FEATURES])


if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = Path(__file__).resolve().parent.parent # Assumes script is in src/
    DATA_DIR = BASE_DIR / 'data' / 'processed'
    MODEL_DIR_PHASE3 = BASE_DIR / 'models' / 'phase3'
    MODEL_DIR_PHASE4 = BASE_DIR / 'models' / 'phase4'
    RESULTS_DIR = BASE_DIR / 'results' / 'phase8'

    # Input data path (assuming phase2 output is the final one used)
    data_path = DATA_DIR / 'phase2_output.csv'
    tfidf_sparse_path = DATA_DIR / 'tfidf_matrix.npz' # Path to the full sparse matrix

    # --- Define Models and Scalers to Analyze ---
    # !! IMPORTANT: Update these paths based on your actual saved best models from phase 3 & 4 !!
    model_paths = {
        # Phase 3 Best Models (Examples - replace with your actual best)
        'P3_tfidf_sparse_LinearSVC': MODEL_DIR_PHASE3 / 'final_tfidf_sparse_LinearSVC.pkl',
        'P3_semantic_DenseNN': MODEL_DIR_PHASE3 / 'final_semantic_Dense_Semantic_Model.h5', # Keras example
        'P3_syntactic_CNN': MODEL_DIR_PHASE3 / 'final_syntactic_CNN1D.joblib', # PyTorch via joblib example
        'P3_lexical_LogReg': MODEL_DIR_PHASE3 / 'final_lexical_LogisticRegression.pkl',

        # Phase 4 Best Fusion Model (Example - replace with your actual best)
        'P4_LateFusion': MODEL_DIR_PHASE4 / 'Late_Fusion_Avg_Prob_final_model_svm.pkl', # Example: Late fusion SVM part
        # Add other parts if needed, e.g., 'P4_LateFusion_NN': MODEL_DIR_PHASE4 / 'Late_Fusion_Avg_Prob_final_model_nn.h5'
        # 'P4_Intermediate_BiLSTM': MODEL_DIR_PHASE4 / 'Intermediate_Sem_Syn_BiLSTM_final_model.h5' # Example if BiLSTM was best
    }

    scaler_paths = {
        # Scalers corresponding to the feature groups used by the models above
        'tfidf_dense': MODEL_DIR_PHASE3 / 'scaler_tfidf_dense.pkl', # Assuming dense tfidf exists
        'lexical': MODEL_DIR_PHASE3 / 'scaler_lexical.pkl',
        'syntactic': MODEL_DIR_PHASE3 / 'scaler_syntactic.pkl',
        'semantic': MODEL_DIR_PHASE3 / 'scaler_semantic.pkl',
        # Add phase 4 scalers if they are different or specific to fusion models
    }

    # --- Analysis Setup ---
    # Define which feature groups each model uses
    model_feature_groups = {
        'P3_tfidf_sparse_LinearSVC': ['tfidf_sparse'],
        'P3_semantic_DenseNN': ['semantic'],
        'P3_syntactic_CNN': ['syntactic'],
        'P3_lexical_LogReg': ['lexical'],
        'P4_LateFusion': ['tfidf_sparse'], # This specific example only needs tfidf for the SVM part
        # 'P4_LateFusion_NN': ['semantic'], # If analyzing the NN part
        # 'P4_Intermediate_BiLSTM': ['semantic', 'syntactic'],
    }

    # --- Run Analysis ---
    analyzer = XAIAnalyzer(model_paths, scaler_paths, data_path, tfidf_sparse_path, output_dir=RESULTS_DIR)

    # Iterate through models and apply appropriate XAI techniques
    for model_name in analyzer.models.keys():
        if model_name not in model_feature_groups:
            print(f"\nWarning: Feature groups not defined for model '{model_name}'. Skipping analysis.")
            continue

        features = model_feature_groups[model_name]
        model_instance = analyzer.models[model_name]
        model_type = type(model_instance).__module__.split('.')[0]

        # --- SHAP Analysis ---
        # KernelExplainer is generally applicable but can be slow.
        # DeepExplainer is faster for TF/Keras non-sequential, PyTorch.
        analyzer.shap_analysis(model_name, features)

        # --- LIME Analysis ---
        # LIME Tabular works for most cases here as we prepare dense data for it.
        analyzer.lime_analysis(model_name, features)

        # --- Integrated Gradients (TensorFlow/Keras only) ---
        if model_type == 'tensorflow' and TF_AVAILABLE:
            analyzer.integrated_gradients_analysis(model_name, features)
        # Add PyTorch IG using Captum here if needed and PYTORCH_AVAILABLE

    # --- Aggregate and Report ---
    analyzer.analyze_feature_categories()
    analyzer.generate_report()

    print(f"\nPhase 8: XAI analysis complete. Results saved to {RESULTS_DIR}")