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





# Task 2: Sentiment Analysis 
## Overview
This script develops and evaluates unigram, bigram, and trigram models for sentiment analysis using the IMDB reviews dataset. These models predict the sentiment of a text as either positive or negative based on the frequency and sequence of words.

## Dataset
The script requires the "IMDB Dataset.csv" which contains movie reviews labeled with sentiments. Ensure that the dataset is in the same directory as the script or modify the `file_path` variable accordingly.

## Features
- **Unigram Model:** Analyzes individual word frequencies to predict sentiment.
- **Bigram Model:** Analyzes frequencies of word pairs to predict sentiment.
- **Trigram Model:** Analyzes frequencies of word triplets to predict sentiment.
- **Performance Metrics:** Calculates precision, recall, F1-score, and overall accuracy for each model.

## Requirements
- Python 3.x
- Pandas
- Collections
- Math

You can install the required libraries using the following command:
```bash
pip install pandas
```

## Usage
Run the script from the command line:
```bash
python sentiment_analysis_models.py
```
The script will execute each model sequentially and display the performance metrics and accuracy for each.

## Performance Metrics
The script outputs the following metrics for each model:

### Unigram Model
- **Precision**: Measure of accuracy considering only the relevant data points.
- **Recall**: Measure of how many truly relevant results are returned.
- **F1-score**: Weighted average of precision and recall.
- **Overall Accuracy**: Proportion of total predictions that were correct.

### Bigram Model
- Similar metrics as the Unigram model, but considers pairs of words.

### Trigram Model
- Similar metrics as the Bigram model, but considers triplets of words.

## Sample Output
## Evaluation Metrics

The following table presents the performance metrics for each sentiment analysis model:

| Model   | Metric     | Negative | Positive |
|---------|------------|----------|----------|
| Unigram | Precision  | 0.8283   | 0.8744   |
|         | Recall     | 0.8822   | 0.8177   |
|         | F1-score   | 0.8544   | 0.8451   |
|         | **Overall Accuracy** | \multicolumn{2}{c|}{0.8499} |
| Bigram  | Precision  | 0.8773   | 0.9019   |
|         | Recall     | 0.9047   | 0.8738   |
|         | F1-score   | 0.8908   | 0.8876   |
|         | **Overall Accuracy** | \multicolumn{2}{c|}{0.8892} |
| Trigram | Precision  | 0.8880   | 0.8993   |
|         | Recall     | 0.9005   | 0.8868   |
|         | F1-score   | 0.8942   | 0.8930   |
|         | **Overall Accuracy** | \multicolumn{2}{c|}{0.8936} |

Note: Precision, Recall, and F1-score are provided for each class ('Negative' and 'Positive'), while the Overall Accuracy represents the overall effectiveness of the model across both classes.


## Conclusion
These models provide a basic framework for sentiment analysis and can be further improved by incorporating more sophisticated natural language processing techniques.

## Future Work
- Implement more complex features like word embeddings.
- Integrate neural networks to improve predictive accuracy.
- Develop a user-friendly GUI for real-time sentiment analysis.

### Instructions
1. Replace `sentiment_analysis_models.py` with the actual filename of your script.
2. Modify the `file_path` in the script or ensure the dataset is in the correct directory.
3. This README is designed to be clear and user-friendly, guiding users through running the script, understanding its functions, and interpreting its outputs.


