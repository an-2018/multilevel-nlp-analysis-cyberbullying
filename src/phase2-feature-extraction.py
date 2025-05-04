import sys
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
# from better_profanity import profanity
from profanity_check import predict, predict_prob
# from profanity_check import predict_prob  # Assuming this is used elsewhere or can be removed if not
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# --- Global Setup ---
# Load models only once

# Patch for the profanity_check library to avoid reloading models
# C:\Users\anils\Desktop\ubi\research\cyberbuling-emojis\source\cyberbullying\venv\lib\site-packages\profanity_check\profanity_check.py
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model for spaCy...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

try:
    sia = SentimentIntensityAnalyzer()
    # Trigger download if VADER lexicon is not found
    sia.polarity_scores('test')
except LookupError:
    import nltk
    print("Downloading VADER lexicon for NLTK...")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# Load the sentence transformer model once
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure 'sentence-transformers' is installed and the model is accessible.")
    sys.exit(1)

# --- Feature Extraction Functions ---

def get_sentence_embedding(text: str) -> np.ndarray:
    """Generates a sentence embedding using a pre-loaded SentenceTransformer model."""
    # Ensure text is a string, handle potential NaN or non-string types
    if not isinstance(text, str):
        text = str(text)  # Convert to string, handles NaN as 'nan'
    return semantic_model.encode(text)

def extract_features(row: pd.Series) -> pd.Series:
    """
    Extracts lexical, syntactic, semantic, and sentiment features from a text row.

    Args:
        row (pd.Series): A row of a DataFrame containing at least a 'clean_text' column.

    Returns:
        pd.Series: The original row augmented with new feature columns.
    """
    text = row.get('clean_text', '')  # Use .get for safety if column might be missing

    # Handle potential NaN or non-string inputs gracefully
    if pd.isna(text) or not isinstance(text, str):
        text = ""  # Treat missing or non-string data as empty string

    words = text.split()
    word_count = len(words)

    # --- Lexical Features ---
    row['word_count'] = word_count
    row['unique_word_ratio'] = len(set(words)) / word_count if word_count > 0 else 0
    row['profanity_score'] = predict_prob([text])[0]

    # --- Syntactic Features (using spaCy) ---
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    for pos_tag, count in pos_counts.items():
        row[f'pos_{pos_tag}'] = count  # Dynamically add POS tag counts

    # --- Semantic Features (Sentence-BERT) ---
    # Ensure the embedding is always a fixed-size array, even for empty text
    embedding = get_sentence_embedding(text)
    # Assuming the model outputs a fixed-size vector (e.g., 384 for all-MiniLM-L6-v2)
    embedding_dim = semantic_model.get_sentence_embedding_dimension()
    if embedding.shape[0] != embedding_dim:
        # Handle potential issues or pad if necessary, though encode usually handles this
        embedding = np.zeros(embedding_dim)

    # Store embedding as a list or tuple for better compatibility with some file formats
    row['semantic_vector'] = embedding.tolist()

    # --- Emotional/Sentiment Features ---
    sentiment = sia.polarity_scores(text)
    for k, v in sentiment.items():
        row[f'sentiment_{k}'] = v

    return row

def process_data(input_path: str, output_path: str, nrows: int = None):
    """
    Loads data, extracts features, and saves the results.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
        nrows (int, optional): Number of rows to read from the input file.
                               If None, reads the entire file. Defaults to None.
    """
    print(f"Loading data from: {input_path}")
    try:
        # Use the nrows parameter in pd.read_csv
        # df = pd.read_csv(input_path, nrows=nrows)
        df = pd.read_csv(input_path)  # Read the full dataset
        # shuffle the DataFrame if nrows is specified
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if nrows:
            print(f"Processing only the first {nrows} rows.")
            # Shuffle the DataFrame and pick the first nrows
            df = df.sample(frac=1, random_state=42).head(nrows).reset_index(drop=True)
            # _, df = train_test_split(df, test_size=0, stratify=df['label'], random_state=42)
        else:
            print(f"Processing all {len(df)} rows.")


    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if 'clean_text' not in df.columns:
        print(f"Error: Input CSV must contain a 'clean_text' column.")
        sys.exit(1)

    print("Extracting features...")
    # Use tqdm for progress bar
    tqdm.pandas(desc="Processing rows")
    df_processed = df.progress_apply(extract_features, axis=1)

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed data to: {output_path}")
    df_processed.to_csv(output_path, index=False)
    print("Processing complete.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract features from text data.")
    # parser.add_argument(
    #     "input_file",
    #     help="Path to the input CSV file (must contain a 'clean_text' column)."
    # )
    # parser.add_argument(
    #     "output_file",
    #     help="Path to save the output CSV file with extracted features."
    # )
    # args = parser.parse_args()
    data_path = Path('data/processed')
    # Ensure the output directory exists
    input_path = data_path / 'phase1_output.csv'
    output_path = data_path / 'phase2_output.csv'
    process_data(input_path, output_path, nrows=None)
