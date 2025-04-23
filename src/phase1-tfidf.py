import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz


def extract_tfidf(texts, max_features=5000):
    """Extracts TF-IDF features from text data."""
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 5),
        max_features=max_features,
        sublinear_tf=True,
        analyzer='word',
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def extract_tfidf_features(df, tfidf_max_features=5000):
    print("Extracting TF-IDF features...")
    try:
        # Ensure 'clean_text' column exists
        if 'clean_text' not in df.columns:
            print("Error: 'clean_text' column is required for TF-IDF feature extraction.")
            sys.exit(1)
        train_texts = df['clean_text'].astype(str)
        vectorizer, tfidf_matrix = extract_tfidf(train_texts, max_features=tfidf_max_features)

        # Convert sparse matrix to DataFrame (if needed) or store separately
        # df['tfidf_feature'] = [row.toarray().flatten() for row in tfidf_matrix]
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Saving Vectorizer and Matrix to file
        save_npz('data/processed/tfidf_matrix.npz', tfidf_matrix)
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        print("TF-IDF features extracted and added to DataFrame.")

        return df, vectorizer, tfidf_matrix
    except Exception as e:
        print(f"Error extracting TF-IDF features: {e}")
        sys.exit(1)

def phase1_tfidf(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    result_path = Path('results/phase1')
    # Ensure the output directory exists
    result_path.mkdir(parents=True, exist_ok=True)

    # Preprocess text
    df['clean_text'] = df['text'].str.lower().str.replace(r'http\S+', '', regex=True)

    df, vectorizer, tfidf_matrix = extract_tfidf_features(df)
    # Compute TF-IDF for n-grams (uni-, bi-, tri-grams)
    # vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    # tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    # ----- Aproach 1: Save TF-IDF matrix as a sparse matrix -----
    # # Convert to DataFrame and merge with original dataset
    # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    # df = pd.concat([df, tfidf_df], axis=1)
    #
    # # Visualize top 50 terms
    # wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(
    #     dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).flatten().tolist()[0]))
    # )
    # plt.imshow(wordcloud)
    # plt.savefig(result_path / 'wordcloud.png')
    #
    # Save TF-IDF-augmented dataset
    # df.to_csv(output_path, index=False)
    # ----- Aproach 2: Save TF-IDF matrix as a dense matrix -----
    # Compute TF-IDF matrix
    # vectorizer = TfidfVectorizer(ngram_range=(2, 5), max_features=5000)
    # tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    # visualize top 50 terms
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(
        dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).flatten().tolist()[0]))
    )
    plt.imshow(wordcloud)

    plt.axis('off')
    plt.savefig(result_path / 'wordcloud.png')
    plt.close()
    # Save TF-IDF matrix as a dense matrix
    # Convert to DataFrame and merge with original dataset
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{col}" for col in vectorizer.get_feature_names_out()])

    # Store TF-IDF as sparse matrix
    tfidf_sparse = csr_matrix(tfidf_matrix)

    # --- Size Comparison (Corrected) ---
    # Calculate the hypothetical size if it were a dense matrix
    dense_size_bytes = tfidf_matrix.shape[0] * tfidf_matrix.shape[1] * tfidf_matrix.dtype.itemsize
    # Calculate the actual size of the sparse matrix components
    # data stores the non-zero values
    # indices stores the column indices of the non-zero values
    # indptr stores the row pointers
    sparse_size_bytes = tfidf_matrix.data.nbytes + tfidf_matrix.indices.nbytes + tfidf_matrix.indptr.nbytes

    # Save memory usage by ~90% compared to dense arrays
    print(f"Hypothetical Dense size: {dense_size_bytes / 1e6:.1f}MB")
    print(f"Actual Sparse size: {sparse_size_bytes / 1e6:.1f}MB")


    # Store TF-IDF vectors as a single column
    tfidf_df['tfidf'] = [row.toarray().flatten().tolist() for row in tfidf_matrix]

    # Merge with original dataset
    df = pd.concat([df, tfidf_df], axis=1)

    # Save dataset with TF-IDF vectors
    df.to_csv(output_path, index=False)




if __name__ == "__main__":
    data_path = Path('data/processed')
    # Ensure the output directory exists
    data_path.mkdir(parents=True, exist_ok=True)
    # Example usage
    input_path = data_path / 'balanced-merged-data.csv'
    output_path = data_path / 'phase1_output.csv'
    # Execute the function
    phase1_tfidf(input_path, output_path)