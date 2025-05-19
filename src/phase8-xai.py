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
import lime.lime_text
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from tensorflow.keras.models import load_model
from tqdm import tqdm
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    LayerGradientShap,
    Occlusion,
    Saliency,
    GuidedGradCam,
    visualization
)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TF_AVAILABLE = True

# --- Constants ---
RANDOM_STATE = 42
SEMANTIC_DIM = 384
N_SHAP_SAMPLES = 10  # Reduced for speed
N_SHAP_BACKGROUND = 10  # Reduced for speed
N_SHAP_FEATURES = 100  # Limit SHAP to top 100 features by variance
N_LIME_SAMPLES = 5
N_LIME_FEATURES = 10
N_IG_SAMPLES = 50
N_CAPTUM_SAMPLES = 10  # Number of samples for Captum analysis

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
        # 'lexical': [col for col in ['word_count', 'unique_word_ratio', 'profanity_score'] if col in df.columns],
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
    """Handles predict_proba for sklearn models, including sparse data and pipelines."""
    # Check if it's a pipeline and the final step needs sparse input
    is_pipeline = hasattr(model, 'steps')
    needs_sparse = False
    if is_pipeline:
        # Check if any step *before* the final predictor requires sparse input
        # This is tricky, but often TfidfVectorizer is involved.
        # A simpler check might be if the input 'data' is already sparse.
        pass # For now, assume data is in the correct format for the pipeline start

    # Use predict_proba if available
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(data)
    # Use decision_function if predict_proba is not available (e.g., LinearSVC)
    elif hasattr(model, 'decision_function'):
        print("Warning: Model lacks predict_proba, using decision_function. LIME/SHAP results might be less interpretable.")
        scores = model.decision_function(data)
        if len(scores.shape) == 1: # Binary case
            # Apply sigmoid to convert scores to pseudo-probabilities
            probs = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs, probs]).T
        else: # Multiclass case (less common here)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Fallback for models like ComplementNB that might only have predict
    elif hasattr(model, 'predict'):
        print("Warning: Model lacks predict_proba and decision_function, using predict. Output will be one-hot encoded.")
        preds = model.predict(data)
        # Convert predictions to pseudo-probabilities (one-hot encoding)
        # Assumes binary classification [class0, class1] and labels are 0 and 1
        num_classes = len(model.classes_) if hasattr(model, 'classes_') else 2
        return np.eye(num_classes)[preds]
    else:
        raise AttributeError(f"Model {type(model)} has neither predict_proba, decision_function, nor predict method.")


def keras_predict_proba(model, data):
    """Prediction function wrapper for Keras models."""
    # Keras predict usually returns shape (n_samples, n_classes) or (n_samples, 1) for binary
    predictions = model.predict(data)
    if predictions.shape[1] == 1: # Binary classification with single output neuron
        # Output is likely sigmoid activation, representing P(class=1)
        probs_class1 = predictions.flatten()
        probs_class0 = 1 - probs_class1
        return np.vstack([probs_class0, probs_class1]).T
    elif predictions.shape[1] > 1: # Multi-class or binary with 2 output neurons (softmax)
        # Assume output represents probabilities for each class
        return predictions
    else: # Should not happen with standard Keras classification outputs
        raise ValueError(f"Unexpected Keras model output shape: {predictions.shape}")


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
        self.explanations = defaultdict(dict)
        
        # Initialize feature names
        self.feature_names = {}
        self.feature_names['lexical'] = self.feature_groups.get('lexical', [])
        self.feature_names['syntactic'] = self.feature_groups.get('syntactic', [])
        self.feature_names['sentiment'] = self.feature_groups.get('sentiment', [])
        self.feature_names['tfidf_dense'] = self.feature_groups.get('tfidf_dense', [])
        self.feature_names['semantic'] = [f'sem_{i}' for i in range(SEMANTIC_DIM)] if 'semantic' in self.feature_groups else []
        self.feature_names['tfidf_sparse'] = self.tfidf_feature_names if self.tfidf_feature_names is not None else \
            ([f'tfidf_{i}' for i in range(self.tfidf_sparse_matrix.shape[1])] if self.tfidf_sparse_matrix is not None else [])

        # Combine dense features for easier access
        self.dense_feature_names = self.feature_names['lexical'] + \
                                 self.feature_names['syntactic'] + \
                                 self.feature_names['sentiment'] + \
                                 self.feature_names['tfidf_dense']
        
        # Initialize Captum attributes
        self.captum_attributors = {}
        self._initialize_captum_attributors()

    def _initialize_captum_attributors(self):
        """Initialize Captum attributors for PyTorch models."""
        for model_name, model in self.models.items():
            if isinstance(model, nn.Module):
                self.captum_attributors[model_name] = {
                    'integrated_gradients': IntegratedGradients(model),
                    'saliency': Saliency(model),
                    'occlusion': Occlusion(model),
                    'gradient_shap': LayerGradientShap(model, model.get_input_layer())
                }

    def analyze_sample_with_captum(self, model_name, sample_idx, feature_group_keys):
        """Analyze a single sample using Captum attribution methods."""
        if model_name not in self.captum_attributors:
            print(f"Captum analysis not available for model {model_name}")
            return

        # Prepare sample data
        sample_data = self._prepare_data_for_model(feature_group_keys, [sample_idx], scale=True)
        if isinstance(sample_data, list):
            sample_data = [torch.tensor(d, dtype=torch.float32) for d in sample_data]
        else:
            sample_data = torch.tensor(sample_data, dtype=torch.float32)

        # Get model prediction
        model = self.models[model_name]
        model.eval()
        with torch.no_grad():
            prediction = model(sample_data)
            pred_class = torch.argmax(prediction).item()

        # Calculate attributions using different methods
        attributions = {}
        for method_name, attributor in self.captum_attributors[model_name].items():
            if method_name == 'integrated_gradients':
                attributions[method_name] = attributor.attribute(
                    sample_data,
                    target=pred_class,
                    n_steps=50
                )
            elif method_name == 'saliency':
                attributions[method_name] = attributor.attribute(
                    sample_data,
                    target=pred_class
                )
            elif method_name == 'occlusion':
                attributions[method_name] = attributor.attribute(
                    sample_data,
                    target=pred_class,
                    strides=(3,),
                    sliding_window_shapes=(3,)
                )

        # Visualize attributions
        self._visualize_captum_attributions(
            model_name,
            sample_idx,
            attributions,
            feature_group_keys
        )

    def _visualize_captum_attributions(self, model_name, sample_idx, attributions, feature_group_keys):
        """Visualize Captum attributions for a sample."""
        sample_text = self.df.iloc[sample_idx]['clean_text']
        true_label = self.df.iloc[sample_idx]['label']
        
        # Create visualization directory
        vis_dir = self.output_dir / 'captum_visualizations' / model_name
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Plot attributions for each method
        for method_name, attr in attributions.items():
            plt.figure(figsize=(15, 5))
            
            # Convert attributions to numpy if they're tensors
            if isinstance(attr, torch.Tensor):
                attr = attr.detach().cpu().numpy()
            
            # Plot feature importance
            feature_names = self._get_feature_names(feature_group_keys)
            sorted_idx = np.argsort(np.abs(attr))
            plt.barh(range(len(sorted_idx)), np.abs(attr[sorted_idx]))
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title(f'{method_name.capitalize()} Attribution for Sample: "{sample_text[:50]}..."')
            plt.xlabel('Attribution Magnitude')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(vis_dir / f'sample_{sample_idx}_{method_name}.png')
            plt.close()

    def analyze_linguistic_patterns(self, model_name, feature_group_keys, num_samples=5):
        """Analyze linguistic patterns that contribute to classification."""
        # Select diverse samples from each class
        samples = self._select_diverse_samples(num_samples)
        
        for sample_idx in samples:
            # Get sample text and true label
            sample_text = self.df.iloc[sample_idx]['clean_text']
            true_label = self.df.iloc[sample_idx]['label']
            
            # Perform LIME analysis
            lime_exp = self.lime_analysis(
                model_name,
                feature_group_keys,
                num_samples=1,
                num_features=20,
                sample_indices=[sample_idx]
            )
            
            # Visualize linguistic patterns
            self._visualize_linguistic_patterns(
                model_name,
                sample_idx,
                sample_text,
                true_label,
                lime_exp
            )

    def _visualize_linguistic_patterns(self, model_name, sample_idx, text, true_label, lime_exp):
        """Visualize linguistic patterns that contribute to classification."""
        vis_dir = self.output_dir / 'linguistic_patterns' / model_name
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Word-level importance
        words = text.split()
        word_importance = [abs(exp[1]) for exp in lime_exp[0]]
        ax1.bar(range(len(words)), word_importance)
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45)
        ax1.set_title(f'Word-level Importance for: "{text[:50]}..."')
        ax1.set_ylabel('Importance Score')
        
        # Plot 2: Feature category importance
        feature_categories = defaultdict(float)
        for exp in lime_exp[0]:
            feature_name = exp[0]
            importance = abs(exp[1])
            category = self._get_feature_category(feature_name)
            feature_categories[category] += importance
        
        categories = list(feature_categories.keys())
        values = list(feature_categories.values())
        ax2.bar(categories, values)
        ax2.set_title('Feature Category Importance')
        ax2.set_ylabel('Cumulative Importance')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'sample_{sample_idx}_linguistic_patterns.png')
        plt.close()

    def _get_feature_category(self, feature_name):
        """Determine the category of a feature based on its name."""
        if feature_name.startswith('pos_'):
            return 'Syntactic'
        elif feature_name.startswith('sentiment_'):
            return 'Sentiment'
        elif feature_name.startswith('tfidf_'):
            return 'Lexical'
        elif feature_name == 'semantic_vector':
            return 'Semantic'
        else:
            return 'Other'

    def _select_diverse_samples(self, num_samples):
        """Select diverse samples from each class for analysis."""
        samples = []
        for label in self.df['label'].unique():
            class_samples = self.df[self.df['label'] == label].index
            if len(class_samples) > 0:
                # Select samples with different lengths and characteristics
                selected = np.random.choice(class_samples, min(num_samples, len(class_samples)), replace=False)
                samples.extend(selected)
        return samples

    def _load_models(self, model_paths):
        """Load trained models from different phases"""
        models = {}
        print("\nLoading models...")
        for name, path in model_paths.items():
            path = Path(path) # Ensure path is a Path object
            if not path.exists():
                print(f"  Warning: Model file not found for '{name}' at {path}. Skipping.")
                continue
            try:
                if path.suffix == '.pkl':
                    models[name] = joblib.load(path)
                    print(f"  Loaded sklearn/joblib model '{name}' from {path}")
                elif path.suffix == '.h5' and TF_AVAILABLE:
                    # Wrap Keras loading in a function to handle potential custom objects if needed
                    def load_keras_model(filepath):
                        # Add custom_objects if your model uses custom layers/losses/etc.
                        # return load_model(filepath, custom_objects={'CustomLayer': CustomLayer})
                        return load_model(filepath)
                    models[name] = load_keras_model(path)
                    print(f"  Loaded Keras model '{name}' from {path}")
                # Add PyTorch loading if needed
                # elif path.suffix in ['.pt', '.pth']:
                #     # ... (PyTorch loading logic) ...
                else:
                    print(f"  Warning: Unsupported model file type '{path.suffix}' for '{name}'. Skipping.")

            except Exception as e:
                print(f"  Error loading model '{name}' from {path}: {e}")
                # Optionally, remove the failed model entry
                if name in models:
                    del models[name]

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
        """Get the appropriate prediction function wrapper for the model."""
        if model_name not in self.models:
            print(f"Warning: Model '{model_name}' not loaded.")
            return None

        model = self.models[model_name]
        model_module = type(model).__module__

        # Check for scikit-learn models
        # Includes common classifiers and checks for joblib persistence artifacts
        if model_module.startswith('sklearn') or 'sklearn' in str(type(model)) or isinstance(model, (SVC, LinearSVC, LogisticRegression, RandomForestClassifier, ComplementNB)):
            # Return a lambda that calls the sklearn_predict_proba helper
            return lambda data: sklearn_predict_proba(model, data)

        # Check for Keras models (TensorFlow backend)
        elif TF_AVAILABLE and ('tensorflow.python.keras' in model_module or 'keras.src.models' in model_module or isinstance(model, tf.keras.Model)):
            # Return a lambda that calls the keras_predict_proba helper
            return lambda data: keras_predict_proba(model, data)

        # Check for PyTorch models (if PyTorch support is added)
        # elif PYTORCH_AVAILABLE and ('torch.nn.modules' in model_module or isinstance(model, torch.nn.Module)):
        #     return lambda data: pytorch_predict_proba(model, data)

        else:
            print(f"Warning: Unsupported model type ('{model_module}') or missing library for {model_name}. Cannot create prediction function.")
            return None


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
        if sample_indices is not None:
            df_subset = self.df.iloc[sample_indices].reset_index(drop=True)
            if self.tfidf_sparse_matrix is not None:
                tfidf_sparse_subset = self.tfidf_sparse_matrix[sample_indices]
            else:
                tfidf_sparse_subset = None
        else:
            df_subset = self.df
            tfidf_sparse_subset = self.tfidf_sparse_matrix


        prepared_parts = []
        feature_names_list = []
        requires_sparse = False
        requires_dense = False # Track if any dense features are included

        for key in feature_group_keys:
            if key == 'tfidf_sparse':
                if tfidf_sparse_subset is None:
                    print(f"Warning: Requested tfidf_sparse but matrix not loaded or sampled.")
                    continue
                prepared_parts.append(tfidf_sparse_subset)
                feature_names_list.extend(self.feature_names.get('tfidf_sparse', [])) # Use .get for safety
                requires_sparse = True
            elif key == 'semantic':
                if 'semantic' not in self.feature_groups:
                    print(f"Warning: Requested semantic features but not found in data.")
                    continue
                # Ensure semantic vectors are stacked correctly
                try:
                    data_raw = np.stack(df_subset['semantic_vector'].values)
                except ValueError as e:
                    print(f"Error stacking semantic vectors: {e}. Check for inconsistent shapes or empty arrays.")
                    # Handle error: maybe skip this feature group or the entire data prep
                    return None, [] # Indicate failure

                scaler = self.scalers.get('semantic')
                if scaler and scale:
                    data_scaled = scaler.transform(data_raw)
                    prepared_parts.append(data_scaled)
                else:
                    if scale and not scaler: print(f"Warning: Scaler for '{key}' not found. Using unscaled data.")
                    prepared_parts.append(data_raw)
                feature_names_list.extend(self.feature_names.get('semantic', []))
                requires_dense = True
            else: # Dense features (lexical, syntactic, sentiment, tfidf_dense)
                cols = self.feature_groups.get(key)
                if not cols:
                    print(f"Warning: Feature group '{key}' not found or empty.")
                    continue

                # Check if all columns exist in the dataframe subset
                missing_cols = [col for col in cols if col not in df_subset.columns]
                if missing_cols:
                    print(f"Warning: Columns {missing_cols} for group '{key}' not found in DataFrame. Skipping these.")
                    cols = [col for col in cols if col in df_subset.columns]
                    if not cols: continue # Skip group if no columns remain

                data_raw = df_subset[cols].values.astype(np.float32)
                # Handle potential NaNs
                if np.isnan(data_raw).any():
                    print(f"Warning: NaNs found in group '{key}'. Imputing with 0.")
                    data_raw = np.nan_to_num(data_raw)

                scaler = self.scalers.get(key)
                # print(f"Preparing data for model: {model_name}, features: {feature_type}")
                # Add this line if data_raw is a pandas DataFrame
                if hasattr(data_raw, 'columns'):
                    print(f"Columns in data_raw: {data_raw.columns.tolist()}")
                # Add this line to see the shape
                print(f"Shape of data_raw: {data_raw.shape}")
                # Add this to see what the scaler expects
                print(f"Scaler expects {scaler.n_features_in_} features")
                if scaler and scale:
                    data_scaled = scaler.transform(data_raw)
                    prepared_parts.append(data_scaled)
                else:
                    if scale and not scaler: print(f"Warning: Scaler for '{key}' not found. Using unscaled data.")
                    prepared_parts.append(data_raw)
                feature_names_list.extend(self.feature_names.get(key, []))
                requires_dense = True

        if not prepared_parts:
            print("Error: No data parts were prepared.")
            return None, []

        # Combine features
        if len(prepared_parts) == 1:
            final_data = prepared_parts[0]
            # Convert to dense if requested AND it's currently sparse
            if requires_sparse and return_dense_array:
                try:
                    final_data = final_data.toarray()
                    print(f"  Converted sparse data part to dense array (shape: {final_data.shape})")
                except MemoryError:
                    print("MemoryError: Cannot convert sparse matrix to dense array. Try reducing sample size or using sparse-aware methods.")
                    return None, []
        else:
            # Check if sparse matrix is involved
            sparse_indices = [i for i, part in enumerate(prepared_parts) if hasattr(part, 'tocsr') or hasattr(part, 'tocsc')] # Check for sparse types
            if len(sparse_indices) > 1:
                print("Error: Cannot directly combine multiple sparse matrices with dense ones using hstack. Revise strategy.")
                return None, []
            elif len(sparse_indices) == 1 and requires_dense: # Combine one sparse and one or more dense
                sparse_idx = sparse_indices[0]
                sparse_mat = prepared_parts.pop(sparse_idx)
                # Combine all remaining (dense) parts
                try:
                    dense_mat = np.hstack([part for i, part in enumerate(prepared_parts) if i not in sparse_indices])
                except ValueError as e:
                    print(f"Error during dense hstack: {e}. Shapes: {[p.shape for i, p in enumerate(prepared_parts) if i not in sparse_indices]}")
                    return None, []


                # Combine sparse and dense using sparse hstack
                from scipy.sparse import hstack as sparse_hstack
                try:
                    # Ensure consistent format, CSR is often preferred
                    final_data = sparse_hstack([sparse_mat.tocsr(), dense_mat], format='csr')
                    print(f"  Combined sparse and dense features (shape: {final_data.shape})")
                except ValueError as e:
                    print(f"Error combining sparse and dense: {e}. Shapes: {sparse_mat.shape}, {dense_mat.shape}")
                    return None, []

                # Adjust feature names order if sparse wasn't first
                if sparse_idx != 0:
                    # This logic needs careful checking based on how hstack orders things
                    # Assuming sparse comes first in the final hstack result
                    sparse_names = self.feature_names.get('tfidf_sparse', [])
                    other_names = [name for i, key in enumerate(feature_group_keys) if key != 'tfidf_sparse' for name in self.feature_names.get(key, [])]
                    feature_names_list = sparse_names + other_names
                    print("  Adjusted feature name order for combined sparse/dense.")


                if return_dense_array:
                    try:
                        final_data = final_data.toarray()
                        print(f"  Converted combined sparse matrix to dense array (shape: {final_data.shape})")
                    except MemoryError:
                        print("MemoryError: Cannot convert combined sparse matrix to dense array.")
                        return None, []
            elif requires_sparse: # Only sparse parts (should be just one based on above check)
                final_data = prepared_parts[0]
                if return_dense_array:
                    try:
                        final_data = final_data.toarray()
                        print(f"  Converted sparse-only data to dense array (shape: {final_data.shape})")
                    except MemoryError:
                        print("MemoryError: Cannot convert sparse matrix to dense array.")
                        return None, []
            else: # All parts are dense
                try:
                    final_data = np.hstack(prepared_parts)
                    print(f"  Combined dense features (shape: {final_data.shape})")
                except ValueError as e:
                    print(f"Error during dense hstack: {e}. Shapes: {[p.shape for p in prepared_parts]}")
                    return None, []

        # Final check on feature names length
        if final_data is not None and final_data.shape[1] != len(feature_names_list):
            print(f"Error: Final data columns ({final_data.shape[1]}) do not match feature names count ({len(feature_names_list)}).")
            # print(f"DEBUG: Feature names list: {feature_names_list[:10]}...{feature_names_list[-10:]}") # Debug print
            return None, []


        return final_data, feature_names_list


    def shap_analysis(self, model_name, feature_group_keys, sample_size=N_SHAP_SAMPLES, background_size=N_SHAP_BACKGROUND):
        """Perform SHAP analysis on selected model using specified features."""
        print(f"\n--- Performing SHAP analysis for {model_name} ({', '.join(feature_group_keys)}) ---")
        if model_name not in self.models:
            print(f"Model '{model_name}' not loaded. Skipping SHAP.")
            return

        model = self.models[model_name]
        predict_fn_base = self._get_predict_fn(model_name)
        if predict_fn_base is None:
            print("Cannot get prediction function. Skipping SHAP.")
            return

        # Prepare data (full feature set)
        data, feature_names = self._prepare_data_for_model(
            feature_group_keys,
            scale=True, # Scaling is generally recommended for SHAP
            return_dense_array=True # Always use dense for SHAP
        )
        if data is None:
            print("Data preparation failed. Skipping SHAP.")
            return

        num_available_samples = data.shape[0]
        actual_sample_size = min(sample_size, num_available_samples)
        if actual_sample_size < sample_size:
            print(f"Warning: Requested sample size {sample_size} > available data {num_available_samples}. Using {actual_sample_size} samples.")
        sample_indices = np.random.choice(num_available_samples, actual_sample_size, replace=False)
        data_sample = data[sample_indices]

        actual_background_size = min(background_size, num_available_samples)
        available_indices_for_bg = np.setdiff1d(np.arange(num_available_samples), sample_indices)
        if len(available_indices_for_bg) >= actual_background_size:
            background_indices = np.random.choice(available_indices_for_bg, actual_background_size, replace=False)
        elif num_available_samples > 0:
            print("Warning: Not enough non-overlapping data for background. Using random sample which may include explained instances.")
            background_indices = np.random.choice(num_available_samples, actual_background_size, replace=False)
        else:
            print("Error: No data available for background set.")
            return
        background_data = data[background_indices]

        # --- Choose SHAP explainer ---
        explainer = None
        shap_values = None
        expected_value = None # Store base value

        try:
            # 1. TreeExplainer (for RF, XGBoost) - Handles dense/sparse automatically if model supports it
            if isinstance(model, (RandomForestClassifier, XGBClassifier)) and not hasattr(data, 'tocsr'): # TreeExplainer prefers dense
                print("  Using TreeExplainer.")
                explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional") # Use background for interventional
                shap_values = explainer.shap_values(data_sample) # Check_additivity=False might be needed sometimes
                expected_value = explainer.expected_value

            # 2. DeepExplainer (for TensorFlow/Keras)
            elif TF_AVAILABLE and ('tensorflow.python.keras' in type(model).__module__ or 'keras.src.models' in type(model).__module__ or isinstance(model, tf.keras.Model)):
                print("  Using DeepExplainer.")
                # DeepExplainer expects tensors and dense data
                try:
                    background_tensor = tf.convert_to_tensor(background_data, dtype=tf.float32)
                    data_sample_tensor = tf.convert_to_tensor(data_sample, dtype=tf.float32)
                    explainer = shap.DeepExplainer(model, background_tensor)
                    shap_values = explainer.shap_values(data_sample_tensor)
                    expected_value = explainer.expected_value
                except Exception as deep_err:
                    print(f"  DeepExplainer failed: {deep_err}. Falling back to KernelExplainer.")
                    # Fallback will happen in the next 'if' block

            # 3. LinearExplainer (for Linear models, handles sparse)
            elif isinstance(model, (LinearSVC, LogisticRegression, ComplementNB)) and not hasattr(data, 'tocsr'):
                print("  Using LinearExplainer.")
                # LinearExplainer needs coefficients and potentially intercept
                # It might require specific data format (dense or sparse) depending on implementation
                # Simplest form: provide model and data (often background)
                try:
                    # May need feature_perturbation='interventional' and background data
                    explainer = shap.LinearExplainer(model, background_data, feature_perturbation="interventional")
                    shap_values = explainer.shap_values(data_sample)
                    expected_value = explainer.expected_value
                except Exception as linear_err:
                    print(f"  LinearExplainer failed: {linear_err}. Falling back to KernelExplainer.")
                    # Fallback will happen below

            # 4. KernelExplainer (General fallback)
            if explainer is None:
                print("  Using KernelExplainer (fallback or default).")
                # KernelExplainer needs a function that takes a numpy array (usually dense)
                # and returns probabilities (n_samples, n_classes).
                def wrapped_predict_proba_for_kernel(data_input):
                    # Ensure input is in the format expected by the base predict_fn
                    # If the original data was sparse but we converted to dense for Kernel,
                    # the predict_fn might still expect sparse. This needs careful handling.
                    # For simplicity here, assume predict_fn_base handles the input type correctly.
                    # If predict_fn_base expects sparse, we might need to convert data_input back.
                    # However, LIME/Kernel generally work better with dense representations.

                    # Ensure input is 2D numpy array
                    if not isinstance(data_input, np.ndarray): data_input = np.array(data_input)
                    if data_input.ndim == 1: data_input = data_input.reshape(1, -1)

                    probs = predict_fn_base(data_input) # Shape (n_samples, n_classes)
                    return probs

                # Summarize background data if it's large and dense
                background_summary = background_data
                if not hasattr(data, 'tocsr') and background_data.shape[0] > N_SHAP_BACKGROUND * 2: # Heuristic
                    try:
                        print(f"  Summarizing background data using kmeans ({N_SHAP_BACKGROUND} centers)...")
                        background_summary = shap.kmeans(background_data, N_SHAP_BACKGROUND)
                    except Exception as kmeans_err:
                        print(f"  KMeans background summarization failed: {kmeans_err}. Using random sample.")
                        # Keep background_data as is

                print(f"  Initializing KernelExplainer with background shape: {background_summary.shape}")
                explainer = shap.KernelExplainer(wrapped_predict_proba_for_kernel, background_summary, link="logit") # Use logit link for probabilities

                print(f"  Calculating SHAP values for sample shape: {data_sample.shape}")
                # nsamples='auto' lets SHAP choose based on feature count
                shap_values = explainer.shap_values(data_sample, nsamples='auto', l1_reg="auto")
                expected_value = explainer.expected_value


            # --- Process SHAP results ---
            if shap_values is None:
                print("SHAP values could not be computed.")
                return

            # SHAP values can be a list (one per class) or a single array (for single output models)
            # We typically want the values for the "positive" class (class 1)
            shap_values_positive_class = None
            expected_value_positive_class = None

            if isinstance(shap_values, list) and len(shap_values) == 2: # Common for binary classification
                shap_values_positive_class = shap_values[1]
                if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) == 2:
                    expected_value_positive_class = expected_value[1]
                else:
                    expected_value_positive_class = expected_value # Might be single value if link='logit'
            elif isinstance(shap_values, np.ndarray): # Could be single output or already selected class
                shap_values_positive_class = shap_values
                expected_value_positive_class = expected_value # Assume it corresponds
            else:
                print(f"  Warning: Unexpected SHAP values format: {type(shap_values)}. Cannot reliably extract positive class values.")
                return

            if shap_values_positive_class is None:
                print("Could not extract SHAP values for the positive class.")
                return

            print(f"  SHAP values computed (shape for positive class: {shap_values_positive_class.shape}).")
            self.explanations[model_name]['shap'] = shap_values_positive_class
            self.explanations[model_name]['shap_feature_names'] = feature_names
            self.explanations[model_name]['shap_data_sample'] = data_sample # Store data used
            self.explanations[model_name]['shap_expected_value'] = expected_value_positive_class
            self.explanations[model_name]['shap_sample_original_indices'] = self.df.index[sample_indices].tolist() # Store original DF indices


            # --- Visualization ---
            if shap_values_positive_class is not None:
                # Select top N_SHAP_FEATURES for visualization
                top_n = min(N_SHAP_FEATURES, len(feature_names))
                mean_abs_shap = np.mean(np.abs(shap_values_positive_class), axis=0)
                top_idx = np.argsort(mean_abs_shap)[-top_n:]
                top_idx = top_idx.tolist()  # Ensure it's a list of ints
                # Plot only top N_SHAP_FEATURES
                plt.figure()
                shap.summary_plot(
                    shap_values_positive_class[:, top_idx],
                    data_sample[:, top_idx],
                    feature_names=[feature_names[i] for i in top_idx],
                    max_display=top_n,
                    show=False
                )
                plt.title(f'SHAP Summary Plot for {model_name} (Class 1, Top {top_n})')
                plt.tight_layout()
                save_path = self.output_dir / f'shap_summary_{model_name}.png'
                plt.savefig(save_path)
                print(f"  Saved SHAP summary plot to {save_path}")
                plt.close()

                # Example force plot for the first instance in the sample (top N features)
                try:
                    expl = shap.Explanation(
                        values=shap_values_positive_class[0, top_idx],
                        base_values=expected_value_positive_class,
                        data=data_sample[0, top_idx],
                        feature_names=[feature_names[i] for i in top_idx]
                    )
                    plt.figure()
                    shap.force_plot(
                        expl.base_values,
                        expl.values,
                        expl.data,
                        feature_names=expl.feature_names,
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f'SHAP Force Plot (Instance {sample_indices[0]}) for {model_name} (Top {top_n})')
                    save_path_force = self.output_dir / f'shap_force_instance{sample_indices[0]}_{model_name}.png'
                    plt.savefig(save_path_force, bbox_inches='tight')
                    print(f"  Saved SHAP force plot example to {save_path_force}")
                    plt.close()
                except Exception as force_plot_err:
                    print(f"  Could not generate SHAP force plot: {force_plot_err}")
                    plt.close('all')
            else:
                print("  Skipping force plot example as expected_value was not determined.")


        except Exception as e:
            print(f"  ERROR during SHAP analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
        finally:
            # Explicitly delete large objects and collect garbage
            del data, data_sample, background_data, explainer, shap_values
            if 'shap_values_positive_class' in locals(): del shap_values_positive_class
            if 'background_tensor' in locals(): del background_tensor
            if 'data_sample_tensor' in locals(): del data_sample_tensor
            gc.collect()


    def lime_analysis(self, model_name, feature_group_keys, num_samples=N_LIME_SAMPLES, num_features=N_LIME_FEATURES):
        """Perform LIME analysis on selected model using specified features."""
        print(f"\n--- Performing LIME analysis for {model_name} ({', '.join(feature_group_keys)}) ---")
        if model_name not in self.models:
            print(f"Model '{model_name}' not loaded. Skipping LIME.")
            return

        predict_fn_base = self._get_predict_fn(model_name)
        if predict_fn_base is None:
            print("Cannot get prediction function. Skipping LIME.")
            return

        # LIME Tabular requires dense data.
        # Use unscaled data for the explainer's training_data to learn distributions,
        # but explain instances using scaled data if the model expects it.
        data_unscaled, feature_names = self._prepare_data_for_model(
            feature_group_keys,
            scale=False, # Use unscaled for LIME background/distribution learning
            return_dense_array=True # LIME Tabular needs dense
        )
        if data_unscaled is None:
            print("Unscaled data preparation failed. Skipping LIME.")
            return

        # Prepare scaled data for explaining specific instances (if model uses scaled data)
        data_scaled, _ = self._prepare_data_for_model(
            feature_group_keys,
            scale=True, # Use scaled for explaining instances
            return_dense_array=True
        )
        if data_scaled is None:
            print("Scaled data preparation failed. Skipping LIME.")
            return


        # Check if data preparation was successful and feature names match
        if data_unscaled.shape[1] != len(feature_names) or data_scaled.shape[1] != len(feature_names):
            print(f"Error: Mismatch between data columns and feature names. Unscaled: {data_unscaled.shape[1]}, Scaled: {data_scaled.shape[1]}, Names: {len(feature_names)}. Skipping LIME.")
            return

        # Sample instances to explain (indices from the original dataframe)
        num_available_samples = data_unscaled.shape[0]
        actual_num_samples = min(num_samples, num_available_samples)
        if actual_num_samples < num_samples:
            print(f"Warning: Requested {num_samples} LIME samples, but only {num_available_samples} available. Using {actual_num_samples}.")

        if actual_num_samples == 0:
            print("Error: No data available to sample for LIME. Skipping.")
            return

        sample_indices = np.random.choice(num_available_samples, actual_num_samples, replace=False)

        # Get the corresponding scaled data for the sampled instances
        data_sample_scaled = data_scaled[sample_indices]

        # Create LIME Explainer
        try:
            # Wrap predict_fn for LIME (needs 2D output: [prob_class_0, prob_class_1])
            # This wrapper will receive perturbed *unscaled* data from LIME,
            # so it needs to scale it before passing to the model's predict_fn_base.
            def wrapped_predict_proba_for_lime(data_input_unscaled):
                # Ensure input is 2D numpy array
                if not isinstance(data_input_unscaled, np.ndarray):
                    data_input_unscaled = np.array(data_input_unscaled)
                if data_input_unscaled.ndim == 1:
                    data_input_unscaled = data_input_unscaled.reshape(1, -1)

                # --- Scaling Step within LIME prediction ---
                # Reconstruct the scaling process applied in _prepare_data_for_model
                # This is crucial but complex if multiple scalers are involved.
                # Assuming a single combined scaler or consistent scaling for simplicity.
                # A more robust way is to pass the relevant scaler(s) to this function.
                # For now, let's try applying scalers based on feature groups.
                data_input_scaled = np.zeros_like(data_input_unscaled)
                current_col = 0
                for key in feature_group_keys:
                    if key == 'tfidf_sparse': continue # Sparse handled separately, LIME uses dense

                    cols_in_group = self.feature_names.get(key, [])
                    num_cols_in_group = len(cols_in_group)
                    if num_cols_in_group == 0: continue

                    end_col = current_col + num_cols_in_group
                    data_segment = data_input_unscaled[:, current_col:end_col]

                    scaler = self.scalers.get(key)
                    if scaler:
                        try:
                            # Check if scaler expects the correct number of features
                            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != data_segment.shape[1]:
                                print(f"  LIME Pred Fn Warning: Scaler '{key}' expects {scaler.n_features_in_} features, got {data_segment.shape[1]}. Skipping scaling for this segment.")
                                data_input_scaled[:, current_col:end_col] = data_segment
                            else:
                                data_input_scaled[:, current_col:end_col] = scaler.transform(data_segment)
                        except Exception as scale_err:
                            print(f"  LIME Pred Fn Error scaling segment '{key}': {scale_err}. Using unscaled data for this segment.")
                            data_input_scaled[:, current_col:end_col] = data_segment
                    else:
                        # No scaler found, use unscaled data for this segment
                        data_input_scaled[:, current_col:end_col] = data_segment

                    current_col = end_col

                # Call the base prediction function with the now scaled data
                probs = predict_fn_base(data_input_scaled) # Shape (n_samples, n_classes)
                return probs

            print(f"  Initializing LimeTabularExplainer with training data shape: {data_unscaled.shape}")
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data_unscaled, # Use unscaled data to learn feature distributions
                feature_names=feature_names,
                class_names=['non-harmful', 'harmful'], # Adjust if needed (e.g., based on LabelEncoder)
                mode='classification',
                random_state=RANDOM_STATE,
                # discretize_continuous=True # Consider if features have very different ranges
            )

            print(f"  Explaining {actual_num_samples} instances...")
            lime_explanations = []
            original_df_indices = self.df.index[sample_indices].tolist() # Get original indices from df

            for i in tqdm(range(actual_num_samples)):
                instance_scaled = data_sample_scaled[i] # Explain the scaled instance
                original_df_idx = original_df_indices[i] # Get the original index for logging/saving

                try:
                    exp = explainer.explain_instance(
                        instance_scaled, # Pass the scaled instance here
                        wrapped_predict_proba_for_lime, # Use the wrapper that handles scaling
                        num_features=num_features,
                        top_labels=1 # Explain the predicted class
                    )
                    lime_explanations.append(exp)

                    # Save example visualization
                    if i < 3: # Save first few examples
                        try:
                            # Pass the label used for explanation
                            explained_label_idx = exp.available_labels()[0]
                            fig = exp.as_pyplot_figure(label=explained_label_idx)
                            fig.suptitle(f'LIME Explanation (Instance {original_df_idx}) for {model_name}')
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                            save_path = self.output_dir / f'lime_{model_name}_instance{original_df_idx}.png'
                            fig.savefig(save_path)
                            print(f"    Saved LIME plot for instance {original_df_idx} to {save_path}")
                            plt.close(fig)
                        except IndexError:
                            print(f"    Warning: LIME did not return any labels for instance {original_df_idx}. Cannot save plot.")
                            plt.close('all')
                        except Exception as plot_err:
                            print(f"    Warning: Could not save LIME plot for instance {original_df_idx}. Error: {plot_err}")
                            plt.close('all') # Close any potentially open figures

                except Exception as explain_err:
                    print(f"    Error explaining instance {original_df_idx}: {explain_err}")
                    import traceback
                    traceback.print_exc()


            self.explanations[model_name]['lime'] = lime_explanations
            self.explanations[model_name]['lime_feature_names'] = feature_names # Store names used by LIME
            self.explanations[model_name]['lime_sample_original_indices'] = original_df_indices # Store original indices

            print("  LIME analysis complete.")

        except Exception as e:
            print(f"  ERROR during LIME analysis setup for {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del data_unscaled, data_scaled, data_sample_scaled
            if 'explainer' in locals(): del explainer
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

        # Prepare data - IG needs dense, scaled data typically
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
        num_available_samples = data.shape[0]
        actual_num_samples = min(num_samples, num_available_samples)
        if actual_num_samples < num_samples:
            print(f"Warning: Requested {num_samples} IG samples, but only {num_available_samples} available. Using {actual_num_samples}.")

        if actual_num_samples == 0:
            print("Error: No data available to sample for IG. Skipping.")
            return

        sample_indices = np.random.choice(num_available_samples, actual_num_samples, replace=False)
        data_sample = data[sample_indices]
        original_df_indices = self.df.index[sample_indices].tolist() # Get original indices

        try:
            inputs = tf.convert_to_tensor(data_sample, dtype=tf.float32)

            # Baseline: Use zeros or mean of the *full* dataset (or a large sample)
            # Using zeros is simpler and common.
            baseline_source_data, _ = self._prepare_data_for_model(feature_group_keys, scale=True, return_dense_array=True) # Get scaled data again for baseline calc
            if baseline_source_data is None:
                print("Could not prepare data for baseline calculation. Using zeros.")
                baseline_np = np.zeros_like(data_sample[0:1]) # Shape (1, num_features)
            else:
                baseline_np = np.mean(baseline_source_data, axis=0, keepdims=True) # Mean of scaled data
                print(f"  Using mean of scaled data as baseline (shape: {baseline_np.shape})")
            baseline = tf.convert_to_tensor(baseline_np, dtype=tf.float32)


            m_steps = 50 # Number of steps for approximation
            alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Shape: (m_steps+1,)

            # Interpolate inputs
            # Expand dims for broadcasting: inputs (N, F), baseline (1, F), alphas (S, 1, 1)
            inputs_expanded = tf.expand_dims(inputs, 0)     # Shape: (1, N, F)
            baseline_expanded = tf.expand_dims(baseline, 0) # Shape: (1, 1, F)
            alphas_expanded = tf.reshape(alphas, (m_steps + 1, 1, 1)) # Shape: (S, 1, 1)

            delta = inputs_expanded - baseline_expanded # Shape: (1, N, F)
            interpolated_inputs = baseline_expanded + alphas_expanded * delta # Shape: (S, N, F)

            # Reshape for model prediction if needed (e.g., flatten steps and samples)
            # Original shape: (steps, samples, features)
            # Reshaped: (steps * samples, features)
            num_steps, num_actual_samples, num_features = interpolated_inputs.shape
            interpolated_inputs_reshaped = tf.reshape(interpolated_inputs, (num_steps * num_actual_samples, num_features))

            all_grads = []
            # Process in batches to manage memory
            batch_size = 128 # Adjust based on memory
            num_total_interpolated = interpolated_inputs_reshaped.shape[0]
            num_batches = int(np.ceil(num_total_interpolated / batch_size))

            print(f"  Calculating gradients for {num_total_interpolated} interpolated points...")
            for i in tqdm(range(num_batches)):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_total_interpolated)
                batch_inputs = interpolated_inputs_reshaped[batch_start:batch_end]

                with tf.GradientTape() as tape:
                    tape.watch(batch_inputs)
                    # Assuming model outputs probabilities for class 1 (or logits)
                    # Shape: (batch_size_actual, 1) or (batch_size_actual,)
                    batch_preds = model(batch_inputs, training=False)

                    # Select the output neuron corresponding to the positive class (usually index 1 or the single output)
                    if batch_preds.shape[-1] == 2: # Output shape (batch, 2)
                        target_preds = batch_preds[:, 1]
                    elif batch_preds.shape[-1] == 1: # Output shape (batch, 1)
                        target_preds = tf.squeeze(batch_preds, axis=-1) # Remove last dim
                    else: # Assume single output, shape (batch,)
                        target_preds = batch_preds

                # Calculate gradients of the target class output w.r.t. the batch inputs
                batch_grads = tape.gradient(target_preds, batch_inputs)

                if batch_grads is None:
                    print(f"Error: Gradients are None for batch {i}. Check model connectivity or input. Ensure the model output used for gradient calculation is connected to the input.")
                    # Try explaining sum of outputs as a fallback?
                    # with tf.GradientTape() as tape:
                    #     tape.watch(batch_inputs)
                    #     batch_preds = model(batch_inputs, training=False)
                    #     target_preds = tf.reduce_sum(batch_preds, axis=-1) # Sum outputs
                    # batch_grads = tape.gradient(target_preds, batch_inputs)
                    # if batch_grads is None:
                    #      print("Error: Gradients still None even after summing outputs. Cannot proceed with IG.")
                    #      return
                    # else:
                    #      print("Warning: Using sum of outputs for IG gradient calculation.")
                    return # Stop if gradients fail

                all_grads.append(batch_grads.numpy())

            if not all_grads:
                print("Error: No gradients were computed.")
                return

            # Concatenate gradients and reshape back to (steps, samples, features)
            grads_concat = np.concatenate(all_grads, axis=0)
            grads_tensor = np.reshape(grads_concat, (num_steps, num_actual_samples, num_features))

            # Average gradients using trapezoidal rule approximation
            # avg_grads = (grads_tensor[:-1] + grads_tensor[1:]) / 2.0 # Grads for intervals
            # avg_grads = np.mean(avg_grads, axis=0) # Average over steps -> (samples, features)
            # Simpler mean approximation:
            avg_grads = np.mean(grads_tensor, axis=0) # Average over steps -> (samples, features)


            # Calculate Integrated Gradients: (input - baseline) * avg_gradient
            input_minus_baseline = (inputs.numpy() - baseline.numpy()) # Shape (N, F)
            # Ensure baseline was broadcast correctly if it was (1, F)
            if input_minus_baseline.shape[0] == 1 and data_sample.shape[0] > 1:
                input_minus_baseline = np.repeat(input_minus_baseline, data_sample.shape[0], axis=0)

            integrated_gradients = input_minus_baseline * avg_grads # Shape: (N, F)

            print("  Integrated Gradients computed.")
            self.explanations[model_name]['ig'] = integrated_gradients
            self.explanations[model_name]['ig_feature_names'] = feature_names
            self.explanations[model_name]['ig_data_sample'] = data_sample
            self.explanations[model_name]['ig_sample_original_indices'] = original_df_indices


            # Aggregate and visualize
            attributions = np.mean(np.abs(integrated_gradients), axis=0) # Mean absolute attribution per feature
            top_n = min(20, len(feature_names)) # Show top 20 features or fewer
            if top_n == 0:
                print("  No features to display in IG summary plot.")
                return

            top_indices = np.argsort(attributions)[-top_n:]
            top_attr = attributions[top_indices]
            top_names = [feature_names[i] for i in top_indices]

            plt.figure(figsize=(10, max(6, top_n // 2))) # Adjust figure size
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
            # Clean up TensorFlow resources and large arrays
            del data, data_sample, baseline, inputs
            if 'baseline_source_data' in locals(): del baseline_source_data
            if 'interpolated_inputs' in locals(): del interpolated_inputs
            if 'interpolated_inputs_reshaped' in locals(): del interpolated_inputs_reshaped
            if 'grads_tensor' in locals(): del grads_tensor
            if 'avg_grads' in locals(): del avg_grads
            if 'integrated_gradients' in locals(): del integrated_gradients
            if 'all_grads' in locals(): del all_grads
            gc.collect()
            if TF_AVAILABLE: tf.keras.backend.clear_session()


    def analyze_feature_categories(self):
        """Analyze feature category impacts across models using SHAP or IG."""
        print("\n--- Analyzing Feature Category Impacts ---")
        category_impacts_all_models = {}

        for model_name, explanation_data in self.explanations.items():
            impact_source = None
            values_for_aggregation = None # Use absolute mean values
            f_names = None

            # Prioritize SHAP, then IG
            if 'shap' in explanation_data and 'shap_feature_names' in explanation_data:
                impact_source = 'SHAP'
                shap_values = explanation_data['shap'] # Should be (samples, features) for positive class
                f_names = explanation_data['shap_feature_names']
                if shap_values is not None and shap_values.ndim == 2 and shap_values.shape[0] > 0:
                    values_for_aggregation = np.mean(np.abs(shap_values), axis=0)
                else:
                    print(f"  Warning: Invalid SHAP values shape for {model_name}: {np.shape(shap_values)}. Skipping category analysis.")
                    continue

            elif 'ig' in explanation_data and 'ig_feature_names' in explanation_data:
                impact_source = 'IG'
                ig_values = explanation_data['ig'] # Should be (samples, features)
                f_names = explanation_data['ig_feature_names']
                if ig_values is not None and ig_values.ndim == 2 and ig_values.shape[0] > 0:
                    values_for_aggregation = np.mean(np.abs(ig_values), axis=0)
                else:
                    print(f"  Warning: Invalid IG values shape for {model_name}: {np.shape(ig_values)}. Skipping category analysis.")
                    continue

            # Proceed if we have values and names
            if values_for_aggregation is not None and f_names is not None:
                print(f"  Aggregating category impacts for {model_name} using {impact_source}...")
                if len(values_for_aggregation) != len(f_names):
                    print(f"    Error: Mismatch between {impact_source} values ({len(values_for_aggregation)}) and feature names ({len(f_names)}) for {model_name}. Skipping category analysis.")
                    continue

                # Call the aggregation function
                model_category_impacts = self._aggregate_by_category(values_for_aggregation, f_names)
                if model_category_impacts: # Check if aggregation returned results
                    category_impacts_all_models[model_name] = model_category_impacts
                else:
                    print(f"    Warning: Category aggregation failed for {model_name}.")

            else:
                print(f"  No suitable explanation data (SHAP or IG with features) found for {model_name} to analyze categories.")

        if not category_impacts_all_models:
            print("No category impacts could be calculated for any model.")
            return

        # Visualization
        try:
            # Create DataFrame, ensuring consistent category order
            all_categories = sorted(list(set(cat for impacts in category_impacts_all_models.values() for cat in impacts.keys())))
            df_impacts = pd.DataFrame(category_impacts_all_models).fillna(0).reindex(all_categories) # Use reindex to ensure order

            if not df_impacts.empty:
                ax = df_impacts.plot(kind='bar', figsize=(12, 7), width=0.8)
                plt.title('Feature Category Importance Across Models (Mean Abs Attribution)')
                plt.ylabel('Mean Absolute Attribution Score')
                plt.xlabel('Feature Category')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
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
        if len(values) != len(feature_names):
            print(f"    Error (_aggregate_by_category): Length mismatch! Values: {len(values)}, Names: {len(feature_names)}")
            return {} # Return empty if mismatch

        # Create a mapping from feature name to its index and value
        # Handle potential duplicate feature names by taking the first occurrence's value
        feature_value_map = {}
        seen_names = set()
        for i, name in enumerate(feature_names):
            if name not in seen_names:
                feature_value_map[name] = values[i]
                seen_names.add(name)
            # else: # Optional: Warn about duplicates if needed
            #     print(f"    Debug: Duplicate feature name '{name}' encountered during aggregation.")

        assigned_value_sum = 0.0 # Keep track of assigned impact

        # Iterate through defined feature groups
        for category, defined_features_in_group in self.feature_groups.items():
            category_total_impact = 0.0
            found_count = 0

            if category == 'tfidf_sparse':
                # Use loaded names if available, otherwise fallback to prefix
                target_names = self.feature_names.get('tfidf_sparse', [])
                if not target_names: # Fallback if tfidf_feature_names wasn't loaded
                    target_names = [name for name in feature_value_map if name.startswith('tfidf_')]
                    if 'tfidf_dense' in self.feature_groups:
                        print("    Warning: Using 'tfidf_' prefix for tfidf_sparse fallback. May include dense features.")
            elif category == 'semantic':
                target_names = self.feature_names.get('semantic', [])
                if not target_names: # Fallback if SEMANTIC_DIM wasn't set correctly
                    target_names = [name for name in feature_value_map if name.startswith('sem_')]
            else:
                # For lexical, syntactic, sentiment, tfidf_dense, use the exact names
                target_names = defined_features_in_group

            # Sum impacts for features belonging to this category
            for name in target_names:
                if name in feature_value_map:
                    val = feature_value_map[name]
                    # Fix: Convert numpy array to float if needed
                    if isinstance(val, np.ndarray):
                        if val.ndim == 1 and val.shape[0] == 2:
                            val = float(val[1])  # Use class 1
                        elif val.size == 1:
                            val = float(val.item())
                        else:
                            val = float(np.mean(val))
                    category_total_impact += val
                    assigned_value_sum += val # Add to total assigned
                    found_count += 1
                    # Remove the feature from the map to avoid double counting
                    # and to track unassigned features later
                    del feature_value_map[name]

            if found_count > 0:
                # Fix: Convert numpy array to float if needed
                if isinstance(category_total_impact, np.ndarray):
                    category_total_impact = float(category_total_impact)
                print(f"    Category '{category}': Found {found_count} features. Total Impact: {category_total_impact:.4f}")
                impacts[category] = category_total_impact
            else:
                impacts[category] = 0.0 # Ensure category exists in dict even if 0

        # Handle any remaining features in feature_value_map as 'unassigned'
        unassigned_impact = sum(float(v) if isinstance(v, np.ndarray) else v for v in feature_value_map.values())
        unassigned_count = len(feature_value_map)
        if unassigned_count > 0:
            print(f"    Warning: {unassigned_count} features were not assigned to any category (e.g., {list(feature_value_map.keys())[:10]}...). Total Unassigned Impact: {unassigned_impact:.4f}")
            impacts['unassigned'] = unassigned_impact

        # Sanity check: Total aggregated impact vs total original impact
        total_original_impact = float(np.sum(values))
        total_aggregated_impact = float(sum(impacts.values()))
        if not np.isclose(total_original_impact, total_aggregated_impact):
            print(f"    Warning: Total aggregated impact ({total_aggregated_impact:.4f}) does not match total original impact ({total_original_impact:.4f}). Check for double counting or missed features.")

        return dict(impacts)


    def generate_report(self):
        """Generate comprehensive XAI report summarizing findings."""
        print("\n--- Generating XAI Report ---")
        report_data = []
        summary_insights = []

        for model_name in sorted(self.models.keys()): # Process models alphabetically
            if model_name not in self.explanations:
                print(f"  No explanation data found for {model_name}. Skipping in report.")
                continue

            model_exps = self.explanations[model_name]
            model_report = {'model': model_name}
            summary_insights.append(f"Model: {model_name}")
            summary_insights.append("-" * len(f"Model: {model_name}"))

            top_features_combined = defaultdict(lambda: {'shap': 0.0, 'lime': 0.0, 'ig': 0.0, 'count': 0})

            # --- Process SHAP ---
            if 'shap' in model_exps and 'shap_feature_names' in model_exps:
                shap_values = model_exps['shap'] # Should be (samples, features) for positive class
                f_names = model_exps['shap_feature_names']

                if shap_values is not None and f_names is not None and shap_values.ndim == 2 and len(f_names) == shap_values.shape[1]:
                    shap_values_abs_mean = np.mean(np.abs(shap_values), axis=0)
                    indices = np.argsort(shap_values_abs_mean)[::-1] # Descending order

                    # Get top N features
                    n_top = min(N_LIME_FEATURES, len(f_names)) # Use N_LIME_FEATURES for consistency
                    model_report['shap_top_features'] = {f_names[i]: shap_values_abs_mean[i] for i in indices[:n_top]}

                    summary_insights.append("  Top Features (SHAP - Mean Abs Value):")
                    summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in model_report['shap_top_features'].items()])

                    # Aggregate for combined view
                    for i in indices:
                        top_features_combined[f_names[i]]['shap'] += shap_values_abs_mean[i]
                        top_features_combined[f_names[i]]['count'] += 1

                    # Category impacts from SHAP
                    category_impacts = self._aggregate_by_category(shap_values_abs_mean, f_names)
                    model_report['shap_category_impacts'] = category_impacts
                    summary_insights.append("  Feature Category Impacts (SHAP):")
                    summary_insights.extend([f"    - {cat}: {impact:.4f}" for cat, impact in sorted(category_impacts.items(), key=lambda item: item[1], reverse=True)])
                else:
                    summary_insights.append(f"  SHAP: Data or feature names invalid (Values shape: {np.shape(shap_values)}, Names count: {len(f_names) if f_names else 'None'}).")
            else:
                summary_insights.append("  SHAP: No data or feature names available.")


            # --- Process LIME ---
            if 'lime' in model_exps and model_exps['lime']: # Check if list is not empty
                lime_feature_scores = self._process_lime_explanations(model_exps['lime'])
                model_report['lime_top_features'] = lime_feature_scores
                summary_insights.append(f"\n  Top Features (LIME - Aggregated Abs Score over {len(model_exps['lime'])} instances):")
                summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in lime_feature_scores.items()])
                # Aggregate for combined view
                for feat, score in lime_feature_scores.items():
                    top_features_combined[feat]['lime'] += score
                    top_features_combined[feat]['count'] += 1
            else:
                summary_insights.append("\n  LIME: No explanations available.")

            # --- Process Integrated Gradients ---
            if 'ig' in model_exps and 'ig_feature_names' in model_exps:
                ig_values = model_exps['ig'] # Shape (samples, features)
                f_names = model_exps['ig_feature_names']

                if ig_values is not None and f_names is not None and ig_values.ndim == 2 and len(f_names) == ig_values.shape[1]:
                    ig_values_abs_mean = np.mean(np.abs(ig_values), axis=0)
                    indices = np.argsort(ig_values_abs_mean)[::-1]

                    n_top = min(N_LIME_FEATURES, len(f_names))
                    model_report['ig_top_features'] = {f_names[i]: ig_values_abs_mean[i] for i in indices[:n_top]}

                    summary_insights.append("\n  Top Features (Integrated Gradients - Mean Abs Value):")
                    summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in model_report['ig_top_features'].items()])

                    # Aggregate for combined view
                    for i in indices:
                        top_features_combined[f_names[i]]['ig'] += ig_values_abs_mean[i]
                        top_features_combined[f_names[i]]['count'] += 1

                    # Category impacts from IG (only if SHAP wasn't available/valid)
                    if 'shap_category_impacts' not in model_report:
                        category_impacts = self._aggregate_by_category(ig_values_abs_mean, f_names)
                        model_report['ig_category_impacts'] = category_impacts
                        summary_insights.append("  Feature Category Impacts (IG):")
                        summary_insights.extend([f"    - {cat}: {impact:.4f}" for cat, impact in sorted(category_impacts.items(), key=lambda item: item[1], reverse=True)])
                else:
                    summary_insights.append(f"\n  IG: Data or feature names invalid (Values shape: {np.shape(ig_values)}, Names count: {len(f_names) if f_names else 'None'}).")
            else:
                summary_insights.append("\n  IG: No data or feature names available.")


            # --- Calculate combined ranking ---
            # Average score across methods where the feature appeared
            avg_scores = {}
            for feat, d in top_features_combined.items():
                if d['count'] > 0:
                    total_score = d['shap'] + d['lime'] + d['ig'] # Sum scores from all methods
                    avg_scores[feat] = total_score / d['count'] # Average by how many methods found it

            sorted_combined = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)[:N_LIME_FEATURES]
            model_report['combined_top_features'] = dict(sorted_combined)
            summary_insights.append("\n  Top Features (Combined Average Score):")
            if sorted_combined:
                summary_insights.extend([f"    - {feat}: {score:.4f}" for feat, score in sorted_combined])
            else:
                summary_insights.append("    - No features found by any method.")


            report_data.append(model_report)
            summary_insights.append("\n" + "="*50 + "\n")

        # --- Save Reports ---
        # Save detailed report data (convert dicts to strings for CSV)
        try:
            report_df_save = pd.DataFrame(report_data)
            # Convert dictionary columns to string representations for CSV compatibility
            for col in report_df_save.select_dtypes(include=['object']).columns:
                # Check if the first non-NA value is a dict (or list)
                first_val = report_df_save[col].dropna().iloc[0] if not report_df_save[col].dropna().empty else None
                if isinstance(first_val, (dict, list)):
                    report_df_save[col] = report_df_save[col].apply(lambda x: str(x) if pd.notna(x) else None)

            report_path = self.output_dir / 'xai_report_summary.csv'
            report_df_save.to_csv(report_path, index=False)
            print(f"Saved summary XAI report to {report_path}")
        except Exception as e:
            print(f"Error saving summary CSV report: {e}")

        # Save text insights
        insights_path = self.output_dir / 'key_xai_insights.txt'
        try:
            with open(insights_path, 'w', encoding='utf-8') as f: # Specify encoding
                f.write("\n".join(summary_insights))
            print(f"Saved key insights to {insights_path}")
        except Exception as e:
            print(f"Error saving key insights text file: {e}")


    def _process_lime_explanations(self, lime_exps):
        """Extract and aggregate top features from LIME explanations."""
        feature_scores = defaultdict(float)
        total_exps = len(lime_exps)
        if total_exps == 0: return {}

        # Get a combined set of all possible feature names from the analyzer's setup
        # This helps parse LIME's output more reliably.
        all_known_feature_names = set(self.dense_feature_names) | \
                                  set(self.feature_names.get('semantic', [])) | \
                                  set(self.feature_names.get('tfidf_sparse', []))


        for i, exp in enumerate(lime_exps):
            explained_labels = exp.available_labels()
            if not explained_labels:
                print(f"    Warning: No explained label found for LIME instance {i}. Skipping.")
                continue
            label_to_use = explained_labels[0] # Use the first explained label

            try:
                exp_list = exp.as_list(label=label_to_use)
            except KeyError:
                print(f"    Warning: Could not get explanation list for label {label_to_use} in LIME instance {i}. Skipping.")
                continue
            except Exception as e:
                print(f"    Warning: Error getting LIME explanation list for instance {i}: {e}. Skipping.")
                continue


            for feature_condition, score in exp_list:
                # Improved parsing for feature names from LIME's conditions
                feature_name = None
                parts = feature_condition.split(' ') # e.g., ['feature', '<=', 'value'] or ['value', '<', 'feature']

                # Try finding an exact match from known features within the parts
                found_exact = False
                for part in parts:
                    if part in all_known_feature_names:
                        feature_name = part
                        found_exact = True
                        break

                # If no exact match, try a more general approach (less reliable)
                if not found_exact:
                    # Look for the part that is NOT a number or a comparison operator
                    potential_names = [p for p in parts if not p.replace('.','',1).lstrip('-').isdigit() and p not in ['<=', '>', '<', '>=', '==', '!=']]
                    if len(potential_names) == 1:
                        feature_name = potential_names[0]
                    elif len(potential_names) > 1:
                        # Ambiguous, maybe take the first or last? Or log a warning.
                        feature_name = potential_names[0] # Default to first potential name
                        # print(f"    Warning: Ambiguous LIME feature condition '{feature_condition}'. Using '{feature_name}'.")
                    else:
                        # Cannot determine feature name, use the whole condition as a fallback key
                        feature_name = feature_condition
                        # print(f"    Warning: Could not parse feature name from LIME condition '{feature_condition}'. Using full string.")


                if feature_name: # Ensure we got a name
                    feature_scores[feature_name] += abs(score)
                # No need for feature_counts if averaging over total explanations

        # Average score per feature over the total number of explanations analyzed
        avg_feature_scores = {feat: score / total_exps for feat, score in feature_scores.items()}

        # Return top N features based on aggregated average absolute score
        return dict(sorted(avg_feature_scores.items(), key=lambda item: item[1], reverse=True)[:N_LIME_FEATURES])

    def visualize_example_cases(self, model_name, num_cases=4):
        """Selects example cases from the data and visualizes category impacts based on SHAP."""
        print(f"\n--- Visualizing Example Cases for {model_name} (using SHAP) ---")
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
        if model_name not in self.explanations or 'shap' not in self.explanations[model_name]:
            print(f"No SHAP explanation data available for {model_name}. Skipping case visualization.")
            return

        shap_explanation_data = self.explanations[model_name]
        shap_values_sample = shap_explanation_data.get('shap') # (sample_size, features)
        original_indices_sample = shap_explanation_data.get('shap_sample_original_indices') # List of original df indices
        feature_names = shap_explanation_data.get('shap_feature_names')
        data_sample = shap_explanation_data.get('shap_data_sample') # The actual data used for SHAP

        if shap_values_sample is None or original_indices_sample is None or feature_names is None or data_sample is None:
            print("Missing necessary SHAP data (values, indices, names, or data sample). Cannot visualize cases.")
            return

        if len(original_indices_sample) != shap_values_sample.shape[0]:
            print(f"Warning: Mismatch between number of SHAP values ({shap_values_sample.shape[0]}) and original indices ({len(original_indices_sample)}).")
            # Try to proceed if shapes match data_sample
            if data_sample.shape[0] != shap_values_sample.shape[0]:
                print("Error: SHAP values shape doesn't match data sample shape. Cannot proceed.")
                return
            # Use data_sample indices if original_indices are mismatched
            original_indices_sample = self.df.index[shap_explanation_data.get('shap_sample_original_indices')].tolist() # Re-fetch based on stored indices


        # Get predictions for the *sampled* data used in SHAP
        predict_fn = self._get_predict_fn(model_name)
        if predict_fn is None: return

        try:
            # Predict using the *exact data sample* that SHAP used
            pred_probs_sample = predict_fn(data_sample) # Should return (n_samples, n_classes)
            # Ensure probabilities are 2D
            if pred_probs_sample.ndim == 1: # Handle case where only positive class prob is returned
                pred_probs_sample = np.vstack([1 - pred_probs_sample, pred_probs_sample]).T

            preds_sample = np.argmax(pred_probs_sample, axis=1) # Get predicted class index (0 or 1)

        except Exception as e:
            print(f"Prediction failed for SHAP data sample: {e}")
            return

        # Get true labels for the sampled data using the original indices
        try:
            true_labels_sample = self.df.loc[original_indices_sample, 'label_encoded'].values
        except KeyError:
            print("Error: Could not retrieve true labels for the sampled indices. Check 'original_indices_sample'.")
            return


        # Find correct and incorrect predictions within the sample
        correct_mask_sample = (preds_sample == true_labels_sample)
        incorrect_mask_sample = ~correct_mask_sample

        # Select examples from the *sample*
        example_indices_in_sample = []
        # Try to get 2 correct and 2 incorrect
        for mask, label_type in [(correct_mask_sample, 'correct'), (incorrect_mask_sample, 'incorrect')]:
            available_in_sample = np.where(mask)[0] # Indices *within the sample*
            if len(available_in_sample) > 0:
                selected_in_sample = np.random.choice(available_in_sample, size=min(2, len(available_in_sample)), replace=False)
                example_indices_in_sample.extend(selected_in_sample)

        # Limit to num_cases
        example_indices_in_sample = example_indices_in_sample[:num_cases]

        if not example_indices_in_sample:
            print(f"Could not find suitable example cases (correct/incorrect) within the SHAP sample for {model_name}.")
            return

        # Process and plot selected examples
        for sample_idx_pos in example_indices_in_sample:
            original_df_idx = original_indices_sample[sample_idx_pos] # Get the original DataFrame index
            shap_values_for_instance = shap_values_sample[sample_idx_pos] # Get SHAP values for this instance
            true_label_val = true_labels_sample[sample_idx_pos]
            pred_label_val = preds_sample[sample_idx_pos]
            prediction_type = 'correct' if true_label_val == pred_label_val else 'incorrect'

            # Aggregate category impacts for this specific instance using its SHAP values
            category_impacts = self._aggregate_by_category(np.abs(shap_values_for_instance), feature_names) # Use absolute values for impact magnitude

            # Create visualization
            plt.figure(figsize=(10, 7)) # Increased height slightly

            # Plot category impacts
            if category_impacts:
                # Sort categories by absolute impact for plotting
                sorted_categories = sorted(category_impacts.items(), key=lambda item: item[1], reverse=True)
                categories = [item[0] for item in sorted_categories]
                impacts = [item[1] for item in sorted_categories]
                y_pos = np.arange(len(categories))

                plt.barh(y_pos, impacts, align='center', color='skyblue')
                plt.yticks(y_pos, categories)
                plt.xlabel('Mean Absolute SHAP Value Contribution')
                title = (f"Category Impacts for {model_name} (Instance {original_df_idx})\n"
                         f"True: {'harmful' if true_label_val == 1 else 'non-harmful'}, "
                         f"Predicted: {'harmful' if pred_label_val == 1 else 'non-harmful'} "
                         f"({prediction_type.capitalize()})")
                plt.title(title, pad=20) # Add padding to title
                plt.gca().invert_yaxis() # Show most important at top
            else:
                plt.text(0.5, 0.5, 'Category impact aggregation failed.', ha='center', va='center')

            # Add text content below the plot
            try:
                text_content = self.df.loc[original_df_idx, 'clean_text']
                text_str = f"Text: {text_content[:400]}{'...' if len(text_content) > 400 else ''}"
                plt.figtext(0.5, 0.01, text_str, wrap=True, horizontalalignment='center', fontsize=9)
            except KeyError:
                plt.figtext(0.5, 0.01, "Text not found for this index.", wrap=True, horizontalalignment='center', fontsize=9)


            plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # Adjust rect to make more room for text and title
            save_path = self.output_dir / f'case_{model_name}_{original_df_idx}_{prediction_type}.png'
            plt.savefig(save_path)
            plt.close()
            print(f"Saved case visualization: {save_path}")


    def visualize_lime_text(self, model_name, num_samples=3):
        """Generates LIME text explanations IF the model uses TF-IDF sparse features."""
        print(f"\n--- Attempting LIME Text Visualizations for {model_name} ---")

        # 1. Check if the model uses TF-IDF sparse features primarily
        #    (This is a heuristic, LIME text works best when the model directly uses text features)
        model_uses_tfidf_sparse = False
        if model_name in model_feature_groups:
            features_used = model_feature_groups[model_name]
            # Check if 'tfidf_sparse' is the *only* or primary feature group
            if 'tfidf_sparse' in features_used and len(features_used) == 1:
                model_uses_tfidf_sparse = True
            # Add more sophisticated checks if needed (e.g., if TF-IDF is part of a pipeline)

        if not model_uses_tfidf_sparse:
            print(f"  Skipping LIME Text for {model_name}: Model does not appear to directly use sparse TF-IDF features based on config.")
            return

        # 2. Check if necessary components are available
        if model_name not in self.models:
            print(f"  Model {model_name} not found!")
            return
        if self.tfidf_sparse_matrix is None or self.tfidf_feature_names is None:
            print(f"  Skipping LIME Text for {model_name}: Missing sparse TF-IDF matrix or feature names.")
            return
        if 'clean_text' not in self.df.columns:
            print(f"  Skipping LIME Text for {model_name}: 'clean_text' column missing in DataFrame.")
            return

        # 3. Get the model and the correct *vectorizer* (TF-IDF)
        model = self.models[model_name]
        vectorizer = None

        # If the model is a pipeline, try to extract the vectorizer step
        if hasattr(model, 'steps'):
            for step_name, step_obj in model.steps:
                # Check common vectorizer types
                if 'TfidfVectorizer' in str(type(step_obj)):
                    vectorizer = step_obj
                    print(f"  Found TfidfVectorizer step ('{step_name}') in the pipeline.")
                    break
        else:
            # If it's not a pipeline, we assume a vectorizer was used *before* training
            # and we need to load it separately. This path needs to be defined.
            # Example:
            vectorizer_path = MODEL_DIR_PHASE3 / 'tfidf_vectorizer.pkl' # ASSUMED PATH - ADJUST!
            if vectorizer_path.exists():
                try:
                    vectorizer = joblib.load(vectorizer_path)
                    print(f"  Loaded separate TfidfVectorizer from {vectorizer_path}")
                except Exception as e:
                    print(f"  Error loading separate vectorizer from {vectorizer_path}: {e}")
            else:
                print(f"  Warning: Model {model_name} is not a pipeline, and a separate vectorizer was not found at {vectorizer_path}. Cannot use LIME Text.")
                return

        if vectorizer is None:
            print(f"  Error: Could not find or load the TF-IDF vectorizer associated with {model_name}. Cannot use LIME Text.")
            return

        # 4. Create the LIME Text prediction function wrapper
        # This function takes raw text strings and must return prediction probabilities
        def lime_text_predict_proba(text_list):
            # Vectorize the text using the loaded vectorizer
            vectors = vectorizer.transform(text_list) # Output is sparse matrix

            # Get predictions from the model (handle pipeline vs standalone)
            if hasattr(model, 'steps'):
                # If it's a pipeline, the vectorizer is already part of it.
                # However, LIME needs the *full* pipeline prediction.
                # We might need to pass the raw text through the *entire* pipeline.
                # Let's redefine the predict_fn to take text directly if it's a pipeline.
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(text_list)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(text_list)
                    if len(scores.shape) == 1:
                        probs_pos = 1 / (1 + np.exp(-scores))
                        probs = np.vstack([1 - probs_pos, probs_pos]).T
                    else: # Should not happen for binary SVM/LogReg
                        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                else: # Fallback for ComplementNB in pipeline?
                    preds = model.predict(text_list)
                    num_classes = len(model.classes_) if hasattr(model, 'classes_') else 2
                    probs = np.eye(num_classes)[preds]
            else:
                # If model is standalone, use the vectorized data
                base_predict_fn = self._get_predict_fn(model_name)
                if base_predict_fn is None:
                    # Should not happen if we got this far, but safety check
                    raise RuntimeError("Base prediction function not found for LIME Text.")
                probs = base_predict_fn(vectors) # Pass the sparse matrix

            # Ensure shape is (n_samples, n_classes)
            if probs.ndim == 1: # Handle case where only positive class prob is returned
                probs = np.vstack([1 - probs, probs]).T
            elif probs.shape[1] != 2: # Check if it's not (n, 2) for binary
                # This might happen with ComplementNB's one-hot output from our wrapper
                if probs.shape[1] == 1: # Convert (n, 1) sigmoid output
                    probs = np.hstack([1 - probs, probs])
                # Add handling for other unexpected shapes if necessary
                else:
                    print(f"Warning: Unexpected probability shape from model in LIME Text: {probs.shape}")
                    # Attempt to select first two columns if more exist? Risky.
                    if probs.shape[1] > 2: probs = probs[:, :2]
                    # If still not right, return dummy probabilities
                    if probs.shape[1] != 2: return np.ones((len(text_list), 2)) * 0.5


            return probs


        # 5. Get text samples and labels
        texts = self.df['clean_text'].tolist()
        labels = self.df['label_encoded'].tolist() # Assuming 0=non-harmful, 1=harmful
        class_names = ['non-harmful', 'harmful'] # Match label encoding

        # 6. Create LIME text explainer
        try:
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=class_names,
                random_state=RANDOM_STATE
            )
        except Exception as e:
            print(f"  Error initializing LimeTextExplainer: {e}")
            return

        print(f"  Generating LIME Text explanations for {num_samples} random instances...")
        explained_count = 0
        attempts = 0
        max_attempts = num_samples * 5 # Try a few times to find samples

        while explained_count < num_samples and attempts < max_attempts:
            attempts += 1
            idx = np.random.randint(0, len(texts))
            text = texts[idx]
            true_label_idx = labels[idx]

            # Generate explanation
            try:
                num_features = N_LIME_FEATURES # Number of features to show in explanation
                exp = explainer.explain_instance(
                    text_instance=text,
                    classifier_fn=lime_text_predict_proba,
                    top_labels=1, # Explain the top predicted class
                    num_features=num_features # Use N_LIME_FEATURES
                )
                explained_count += 1

                # Save the explanation as HTML
                html_path = self.output_dir / f'lime_text_{model_name}_instance{idx}.html'
                exp.save_to_file(str(html_path))
                print(f"    Saved LIME text explanation for instance {idx} to {html_path}")

                # Optional: Store the explanation object if needed later
                # self.explanations[model_name].setdefault('lime_text', []).append(exp)

            except Exception as e:
                print(f"    Error generating LIME text explanation for instance {idx}: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed debugging

        if explained_count == 0:
            print("  Failed to generate any LIME Text explanations.")


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
    # Using example names from results files provided. Ensure these files exist.
    model_paths = {
        # Phase 3 Best Models (Based on test_set_results.csv)
        'P3_lexical_SVC_RBF': MODEL_DIR_PHASE3 / 'final_lexical_SVC_RBF.pkl',
        'P3_semantic_DenseNN': MODEL_DIR_PHASE3 / 'final_semantic_Dense_Semantic_Model.h5',
        'P3_syntactic_CNN': MODEL_DIR_PHASE3 / 'final_syntactic_CNN_Syntactic_Model.h5', # Assuming .h5 based on name
        'P3_tfidf_dense_LinearSVC': MODEL_DIR_PHASE3 / 'final_tfidf_dense_LinearSVC.pkl',
        'P3_tfidf_sparse_RandomForest': MODEL_DIR_PHASE3 / 'final_tfidf_sparse_RandomForest.pkl', # Using RF from training results

        # Phase 4 Models (Based on test_set_fusion_results.csv)
        'P4_EarlyFusion_SVM': MODEL_DIR_PHASE4 / 'Early__TFIDF_Dense_Lex_Syn____SVM__final_model.pkl', # Renamed for clarity
        # 'P4_Intermediate_BiLSTM': MODEL_DIR_PHASE4 / 'Intermediate_Sem_Syn_BiLSTM_final_model.h5', # Assuming .h5, uncomment if exists
        # 'P4_LateFusion_AvgProb': MODEL_DIR_PHASE4 / 'Late_Fusion_Avg_Prob_final_model_svm.pkl', # Assuming SVM, uncomment if exists
        'P4_Baseline_TFIDF_Sparse_SVM': MODEL_DIR_PHASE4 / 'Baseline__TF_IDF_Sparse____SVM__final_model.pkl', # Renamed for clarity
        # 'P4_Baseline_Semantic_DenseNN': MODEL_DIR_PHASE4 / 'Baseline_Semantic_Dense_NN_final_model.h5' # Assuming .h5, uncomment if exists
    }

    scaler_paths = {
        # Scalers corresponding to the feature groups used by the models above
        'lexical': MODEL_DIR_PHASE3 / 'scaler_lexical.pkl',
        'syntactic': MODEL_DIR_PHASE3 / 'scaler_syntactic.pkl',
        'semantic': MODEL_DIR_PHASE3 / 'scaler_semantic.pkl',
        'tfidf_dense': MODEL_DIR_PHASE3 / 'scaler_tfidf_dense.pkl',
        # Add phase 4 scalers if they are different or specific to fusion models
        # Example: If early fusion used a combined scaler
        'tfidf_dense+lexical+syntactic': MODEL_DIR_PHASE4 / 'scaler_early_fusion_dense_lex_syn.pkl' # Hypothetical path
    }

    # --- Analysis Setup ---
    # Define which feature groups each model uses (CRITICAL for data preparation)
    model_feature_groups = {
        'P3_lexical_SVC_RBF': ['lexical'],
        'P3_semantic_DenseNN': ['semantic'],
        'P3_syntactic_CNN': ['syntactic'],
        'P3_tfidf_dense_LinearSVC': ['tfidf_dense'],
        'P3_tfidf_sparse_RandomForest': ['tfidf_sparse'],

        # Phase 4 Models - Define based on their construction
        'P4_EarlyFusion_SVM': ['tfidf_dense', 'lexical', 'syntactic'], # Based on name 'TFIDF_Dense+Lex+Syn -> SVM'
        # 'P4_Intermediate_BiLSTM': ['semantic', 'syntactic'], # Based on name 'Sem+Syn -> BiLSTM'
        # 'P4_LateFusion_AvgProb': ['???'], # Late fusion needs careful handling - explain base models or the meta-learner
        'P4_Baseline_TFIDF_Sparse_SVM': ['tfidf_sparse'], # Based on name 'TF-IDF Sparse -> SVM'
        # 'P4_Baseline_Semantic_DenseNN': ['semantic'], # Based on name 'Semantic -> Dense NN'
    }

    # --- Run Analysis ---
    analyzer = XAIAnalyzer(model_paths, scaler_paths, data_path, tfidf_sparse_path, output_dir=RESULTS_DIR)

    # Filter model_feature_groups to only include models that were successfully loaded
    model_feature_groups = {name: features for name, features in model_feature_groups.items() if name in analyzer.models}
    print("\n--- Models available for analysis:", list(analyzer.models.keys()))
    print("--- Feature groups defined for analysis:", model_feature_groups)


    # Iterate through successfully loaded models and apply appropriate XAI techniques
    for model_name in analyzer.models.keys():
        if model_name not in model_feature_groups:
            print(f"\nWarning: Feature groups not defined for loaded model '{model_name}'. Skipping analysis.")
            continue

        print(f"\n>>> Analyzing Model: {model_name} <<<")
        features = model_feature_groups[model_name]
        model_instance = analyzer.models[model_name]
        model_module = type(model_instance).__module__
        is_tf_model = TF_AVAILABLE and ('tensorflow.python.keras' in model_module or 'keras.src.models' in model_module or isinstance(model_instance, tf.keras.Model))

        # --- SHAP Analysis ---
        analyzer.shap_analysis(model_name, features)

        # --- LIME Analysis (Tabular) ---
        analyzer.lime_analysis(model_name, features)

        # --- Integrated Gradients (TensorFlow/Keras only) ---
        if is_tf_model:
            analyzer.integrated_gradients_analysis(model_name, features)
        # Add PyTorch IG using Captum here if needed and PYTORCH_AVAILABLE

        # --- LIME Text Analysis (Attempt for TF-IDF Sparse models) ---
        # Check if 'tfidf_sparse' is the primary feature set for this model
        if 'tfidf_sparse' in features and len(features) == 1:
            analyzer.visualize_lime_text(model_name, num_samples=3)


    # --- Aggregate and Report ---
    analyzer.analyze_feature_categories()
    analyzer.generate_report()

    # --- Visualize Example Cases (using SHAP results) ---
    print("\n--- Generating Example Case Visualizations ---")
    for model_name in analyzer.models.keys():
        if model_name in model_feature_groups: # Only visualize for models we could analyze
            analyzer.visualize_example_cases(model_name, num_cases=4)


    print(f"\nPhase 8: XAI analysis complete. Results saved to {RESULTS_DIR}")