# NLP-assignment
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
