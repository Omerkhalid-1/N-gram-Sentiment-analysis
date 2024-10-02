# Task 1: N-gram Language Models

## Introduction
This project involves building simple unigram, bigram, and trigram language models to analyze text data. These models are useful for understanding the structure of language and predicting subsequent words in sentences. This document provides an overview of the models, the dataset, and how to run the application.

## Dataset
The project utilizes the "IMDB Dataset.csv" which consists of movie reviews. The dataset undergoes several preprocessing steps to prepare it for the models:

- Conversion of text to lowercase to maintain consistency.
- Removal of punctuation to focus on words.
- Tokenization of text into individual words.

## Models
### Unigram Model
The unigram model counts the frequency of each individual word in the dataset and identifies the most frequent terms.

### Bigram Model
The bigram model examines pairs of consecutive words to predict the next word based on the previous word.

### Trigram Model
The trigram model extends the prediction capability to consider pairs of words for predicting the third word.

## Requirements
The project requires Python 3 and the following libraries:
- pandas
- string
- collections

These dependencies can be installed using:
```
pip install pandas
```

## Usage
To run the program, follow these steps:

1. Place the "IMDB Dataset.csv" in the same directory as the script.
2. Execute the script using Python:
   ```
   python script_name.py
   ```
3. When prompted, enter:
   - `1` for Unigram analysis.
   - `2` for Bigram predictions.
   - `3` for Trigram predictions.

### Example Commands
- For bigram predictions, after choosing `2`, enter a word to get the likely next word.
- For trigram predictions, after choosing `3`, enter two consecutive words to get the prediction for the third.

## File Structure
```
.
├── IMDB Dataset.csv   # Dataset file
├── script_name.py     # Main script
└── README.md          # This documentation file
```

## Future Work
Enhancements could include:
- Implementing smoothing techniques to handle unseen words.
- Expanding the models to include n-grams of higher order.
- Developing a GUI for easier interaction with the models.

## Conclusion
This project showcases the fundamental techniques in building and utilizing n-gram models for text prediction and analysis. It serves as a practical application of language modeling in the field of natural language processing.
```
