# DATA PREPROCESSING
# Importing the libraries
import csv
import pandas as pd
import string
import math
import pandas as pd
import string
from collections import defaultdict, Counter

def preprocess_text(text):
    lower_text = text.lower()
    clean_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text

def tokenize(text):
    return text.split()

# Buliding Unigram model from the scratch
def unigram_model(data):
    unigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        unigram_counts.update(tokens)
    return unigram_counts

# Build bigram model from the scratch
def bigram_model(data):
    bigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        bigrams = zip(tokens, tokens[1:])
        bigram_counts.update(bigrams)
    return bigram_counts

# Build trigram model from the scratch
def trigram_model(data):
    trigram_counts = Counter()
    for review in data:
        tokens = tokenize(review)
        trigrams = zip(tokens, tokens[1:], tokens[2:])
        trigram_counts.update(trigrams)
    return trigram_counts

# Make prediction using the bigram model

def predict_next_word_bigram(first_word, bigram_model):
    candidates = {pair: count for pair, count in bigram_model.items() if pair[0] == first_word}
    
    if not candidates:
        return None  # Return None if no bigrams found starting with the first word
    most_frequent_bigram = max(candidates, key=candidates.get)
    
    # Return the second word of the most frequent bigram
    return most_frequent_bigram[1]

def predict_trigram_next_word(first_word, second_word, trigram_model):
    # Filter trigrams starting with the given first word
    candidates = {pair: count for pair, count in trigram_model.items() if pair[0] == first_word and pair[1] == second_word}
    
    # Find the most frequent trigram
    if not candidates:
        return None  # Return None if no trigrams found starting with the first word
    most_frequent_trigram = max(candidates, key=candidates.get)
    
    # Return the third word of the most frequent trigram
    return most_frequent_trigram[2]

#//-----------------------------------------------------------------------------//

def main():
    # Load the dataset from a CSV file
    file_path = "IMDB Dataset.csv"
    data = pd.read_csv(file_path)
    
    # Apply text preprocessing to each review in the dataset
    data['processed_review'] = data['review'].apply(preprocess_text)
    
    # Build unigram, bigram, and trigram models
    unigram_model1 = unigram_model(data['processed_review'])
    bigram_model1 = bigram_model(data['processed_review'])
    trigram_model1 = trigram_model(data['processed_review'])
    
    choice = input("Enter the choice (1: Unigram, 2: Bigram, 3: Trigram): ")
    if choice == '1':
         # Display the frequency of each word in the text..
         value_based = {key:value for key, value in sorted(unigram_model1.items(), key=lambda my_dict: my_dict[1], reverse=True)}
         #print(value_based)

    elif choice == '2':
        value_based = {key:value for key, value in sorted(bigram_model1.items(), key=lambda my_dict: my_dict[1], reverse=True)}
        #print(value_based)
        first_word = input(" Enter the first word: ")
        predicted_word = predict_next_word_bigram(first_word, bigram_model1)
        print(f"The predicted next word after {first_word}' '{predicted_word}")

    elif choice == '3':
        value_based = {key:value for key, value in sorted(trigram_model1.items(), key=lambda my_dict: my_dict[1], reverse=True)}
        #print(value_based)
        first_word = input("Enter the first word: ")
        second_word = input("Enter the second word: ")
        predicted_word = predict_trigram_next_word(first_word, second_word, trigram_model1)
        print(f"The predicted next word after '{first_word} {second_word} {predicted_word}'")
    else:
        file_path = "IMDB Dataset.csv"
        data = pd.read_csv(file_path)
        clean_data = preprocess_text(data['review'][0])
        print(clean_data[1][0])

# // -----------------------------------------------------------------------------//
if __name__ == "__main__":
    main()
