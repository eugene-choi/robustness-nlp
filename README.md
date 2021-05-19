# Evaluating Knowledge of Sarcasm in BERT-Style Models
Project for DS-UA 203 - LING-UA 52: Machine Learning for Language Understanding, taught by Prof. Samuel Bowman, Spring 2021.

## Motivation
Sarcasm detection is a challenge for sentiment analysis due to its difference between surface meaning and intended meaning. In this pa- per, we test state-of-the-art models (BERT, RoBERTa) on their grasp on sarcasm using sentiment analysis. We append sarcastic sen- tences from Sarcasm Corpus V2 and SemEval- 2018 to the IMDB and SST2 dataset then ob- serve behavioral changes. We expect predicted labels to become negative or stay negative as sarcasm has an inherent negative sentiment. Our results show that without additional train- ing, BERT-style models lack the knowledge to process sarcasm and its negative sentiment, and manipulating the length and position of the sarcasm concatenation did not show ob- vious trends. Different types of sarcasm and datasets also do not yield notable differences.

## Data Collection
We categorize our datasets into two subcategories: sentiment and sarcasm. Sentiment datasets include IMDB Review Dataset and Stanford Sentiment Treebank 2, while the sarcasm datasets include Sarcasm Corpus V2 and SemEval-2018.

## Modeling and Analysis
Using the BERT-style models, we perform senti- ment analysis on the IMDB Review Dataset and SST2 datasets. First, we get baselines results from the sentiment datasets to ensure the model is work- ing as expected. Then, we perform sentiment analysis on the subset containing only sarcastic data in the Sarcasm Corpus V2 and SemEval-2018 datasets. We change all labels from 1 (marked for sarcasm) to 0 (negative) to fit the task. Then, we collect examples of what the models identify as negative sentiment, then append them to review datasets as perturbations. Perturbation cases are created by length (x < 128 or 128 ≤ x < 256, x = characters) and position (appending at the begin- ning or end of the review text). We designing our tests by referring to the Checklist framework. If models can successfully detect negative sentiment in sarcasm, we expect cases attached to positive reviews to change labels to negative, similar to a directional expectation test, and cases attached to negative reviews to retain the same label of neg- ative, similar to an invariance test. While these combinations may not appear in real life, it can serve as a simulation for when sarcasm is embed- ded in longer contexts.

## Paper
Full paper can be found [here](https://drive.google.com/drive/u/0/my-drive).
