import pandas as pd
import string
from collections import defaultdict, Counter
import math

def preprocess_text(text):
    lower_text = text.lower()
    clean_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text

def tokenize(text):
    return text.split()

def train_test_split(corpus, split_ratio=0.8):
    split_index = int(len(corpus) * split_ratio)
    return corpus[:split_index], corpus[split_index:]

def build_unigram_model(data):
    unigram_counts = defaultdict(Counter)
    class_counts = defaultdict(int)
    
    for _, row in data.iterrows():
        tokens = tokenize(preprocess_text(row['review']))
        class_label = row['sentiment']
        unigram_counts[class_label].update(tokens)
        class_counts[class_label] += len(tokens)
    
    return unigram_counts, class_counts

def build_bigram_model(data):
    """Build bigram count model for each class, correctly counting bigrams."""
    bigram_counts = defaultdict(Counter)
    class_counts = defaultdict(int)
    
    for _, row in data.iterrows():
        tokens = tokenize(preprocess_text(row['review']))
        bigrams = list(zip(tokens, tokens[1:]))  # Convert to list to count and use in update
        class_label = row['sentiment']
        bigram_counts[class_label].update(bigrams)
        class_counts[class_label] += len(bigrams)  # Now we can use len() because bigrams is a list
    
    return bigram_counts, class_counts

def calculate_probabilities1(unigram_counts, class_counts):
    probabilities = {}
    total_vocabulary_size = sum(len(unigram_counts[cls]) for cls in unigram_counts)
    
    for class_label, counts in unigram_counts.items():
        probabilities[class_label] = {}
        for word, count in counts.items():
            probabilities[class_label][word] = (count + 1) / (class_counts[class_label] + total_vocabulary_size)
    
    return probabilities, total_vocabulary_size

def calculate_probabilities2(bigram_counts, class_counts):
    probabilities = {}
    total_vocabulary_size = sum(len(bigram_counts[cls]) for cls in bigram_counts)
    
    for class_label, counts in bigram_counts.items():
        probabilities[class_label] = {}
        for bigram, count in counts.items():
            probabilities[class_label][bigram] = (count + 1) / (class_counts[class_label] + total_vocabulary_size)
    
    return probabilities, total_vocabulary_size

def classify_review1(review, probabilities, class_counts, total_vocabulary_size):
    tokens = tokenize(preprocess_text(review))
    class_scores = {cls: math.log(class_counts[cls] / sum(class_counts.values())) for cls in class_counts}
    
    for token in tokens:
        for cls in probabilities:
            token_prob = probabilities[cls].get(token, 1 / (class_counts[cls] + total_vocabulary_size))
            class_scores[cls] += math.log(token_prob)
    
    return max(class_scores, key=class_scores.get)

def classify_review2(review, probabilities, class_counts, total_vocabulary_size):
    tokens = tokenize(preprocess_text(review))
    bigrams = list(zip(tokens, tokens[1:]))  # Generate bigrams from the review
    class_scores = {cls: math.log(class_counts[cls] / sum(class_counts.values())) for cls in class_counts}
    
    for bigram in bigrams:
        for cls in probabilities:
            bigram_prob = probabilities[cls].get(bigram, 1 / (class_counts[cls] + total_vocabulary_size))
            class_scores[cls] += math.log(bigram_prob)
    
    return max(class_scores, key=class_scores.get)

def build_trigram_model(data):
    """Build trigram count model for each class."""
    trigram_counts = defaultdict(Counter)
    class_counts = defaultdict(int)
    
    for _, row in data.iterrows():
        tokens = ['<s>', '<s>'] + tokenize(preprocess_text(row['review'])) + ['</s>']
        class_label = row['sentiment']
        trigrams = zip(tokens, tokens[1:], tokens[2:])  # Generate trigrams from tokens
        trigram_counts[class_label].update(trigrams)
        class_counts[class_label] += 1
    
    return trigram_counts, class_counts

def calculate_probabilities3(trigram_counts, class_counts):
    """Calculate trigram probabilities for each class."""
    probabilities = {}
    total_vocabulary_size = sum(len(trigram_counts[cls]) for cls in trigram_counts)
    
    for class_label, counts in trigram_counts.items():
        probabilities[class_label] = {}
        for trigram, count in counts.items():
            probabilities[class_label][trigram] = (count + 1) / (class_counts[class_label] + total_vocabulary_size)
    
    return probabilities, total_vocabulary_size

def classify_review3(review, probabilities, class_counts, total_vocabulary_size):
    """Classify review using trigram probabilities."""
    tokens = ['<s>', '<s>'] + tokenize(preprocess_text(review)) + ['</s>']
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))  # Generate trigrams from the review
    class_scores = {cls: math.log(class_counts[cls] / sum(class_counts.values())) for cls in class_counts}
    
    for trigram in trigrams:
        for cls in probabilities:
            trigram_prob = probabilities[cls].get(trigram, 1 / (class_counts[cls] + total_vocabulary_size))
            class_scores[cls] += math.log(trigram_prob)
    
    return max(class_scores, key=class_scores.get)


def compute_metrics(predictions, actuals):
    unique_classes = set(actuals)
    metrics = {}
    
    for cls in unique_classes:
        tp = sum(1 for pred, act in zip(predictions, actuals) if pred == cls and act == cls)
        fp = sum(1 for pred, act in zip(predictions, actuals) if pred == cls and act != cls)
        fn = sum(1 for pred, act in zip(predictions, actuals) if pred != cls and act == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
    
    return metrics

def main():
    print("Unigram Model")
    file_path = "IMDB Dataset.csv"
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, 0.8)

    unigram_counts, class_counts = build_unigram_model(train_data)
    probabilities, total_vocabulary_size = calculate_probabilities1(unigram_counts, class_counts)
    predictions = [classify_review1(row['review'], probabilities, class_counts, total_vocabulary_size) for index, row in test_data.iterrows()]
    actuals = test_data['sentiment'].tolist()

    metrics = compute_metrics(predictions, actuals)
    for cls, cls_metrics in metrics.items():
        print(f"Metrics for class {cls}:")
        for metric, value in cls_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    #-------------- ----------------------------#
    # Classifer for bigram model
    print("\n\nBigram Model")
    bigram_counts, class_counts2 = build_bigram_model(train_data)
    probabilities2, total_vocabulary_size2 = calculate_probabilities2(bigram_counts, class_counts2)
    predictions2 = [classify_review2(row['review'], probabilities2, class_counts2, total_vocabulary_size2) for index, row in test_data.iterrows()]
    actuals2 = test_data['sentiment'].tolist()
    metrics2 = compute_metrics(predictions2, actuals2)
    for cls, cls_metrics in metrics2.items():
        print(f"Metrics for class {cls}:")
        for metric, value in cls_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    accuracy = sum(1 for p, a in zip(predictions2, actuals2) if p == a) / len(actuals)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    #-------------- ----------------------------#

    # Classifer for trigram model
    print("\n\nTrigram Model")
    trigram_counts, class_counts3 = build_trigram_model(train_data)
    probabilities3, total_vocabulary_size3 = calculate_probabilities3(trigram_counts, class_counts3)
    predictions3 = [classify_review3(row['review'], probabilities3, class_counts3, total_vocabulary_size3) for index, row in test_data.iterrows()]
    actuals3 = test_data['sentiment'].tolist()
    metrics = compute_metrics(predictions3, actuals3)
    for cls, cls_metrics in metrics.items():
        print(f"Metrics for class {cls}:")
        for metric, value in cls_metrics.items():
            print(f"{metric}: {value:.4f}")
    accuracy = sum(1 for p, a in zip(predictions3, actuals3) if p == a) / len(actuals)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
