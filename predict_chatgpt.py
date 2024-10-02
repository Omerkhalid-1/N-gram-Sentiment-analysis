import pandas as pd
import string
from collections import defaultdict, Counter

def preprocess_text(text):
    """Convert text to lowercase and remove punctuation."""
    lower_text = text.lower()
    clean_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text

def tokenize(text):
    """Tokenize the text into a list of words."""
    return text.split()

def build_unigram_model(data):
    """Build a unigram model from a pandas Series of text data."""
    unigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        unigram_counts.update(tokens)
    return unigram_counts

def build_bigram_model(data):
    """Build a bigram model from a pandas Series of text data."""
    bigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        bigrams = zip(tokens, tokens[1:])
        bigram_counts.update(bigrams)
    return bigram_counts

def build_trigram_model(data):
    """Build a trigram model from a pandas Series of text data."""
    trigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        trigrams = zip(tokens, tokens[1:], tokens[2:])
        trigram_counts.update(trigrams)
    return trigram_counts

def main():
    # Load the dataset from a CSV file
    file_path = "IMDB Dataset.csv"
    data = pd.read_csv(file_path)
    
    # Apply text preprocessing to each review in the dataset
    data['processed_review'] = data['review'].apply(preprocess_text)
    
    # Build unigram, bigram, and trigram models
    unigram_model = build_unigram_model(data['processed_review'])
    bigram_model = build_bigram_model(data['processed_review'])
    trigram_model = build_trigram_model(data['processed_review'])
    
    # Print some examples from each model to verify their correctness
    print("Unigrams:")
    for unigram, count in unigram_model.most_common(10):
        print(f"{unigram}: {count}")

    print("\nBigrams:")
    for bigram, count in bigram_model.most_common(10):
        print(f"{bigram}: {count}")

    print("\nTrigrams:")
    for trigram, count in trigram_model.most_common(10):
        print(f"{trigram}: {count}")

if __name__ == "__main__":
    main()
