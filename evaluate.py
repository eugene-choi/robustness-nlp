import numpy as np
import torch
import pandas as pd
from sklearn import metrics


def convert_to_numeric_label(predictions):
  if(predictions == 'ENTAILMENT'):
    return 0
  elif(predictions == 'CONTRADICTION'):
    return 1
  elif(predictions == 'NEUTRAL'):
    return 2


def evaluate(passage, pipe):
  predictions = []
  prediction_scores = []
  for tweet in passage:
    output = pipe(tweet)
    predictions.append(convert_to_numeric_label(output[0]['label']))
    prediction_scores.append(output[0]['score'])
  return predictions, prediction_scores


def plot_scores(predictions, prediction_scores, label):
  label_scores = []
  for i in range(len(predictions)):
    if(predictions[i] == label):
      label_scores.append(prediction_scores[i])
  s = pd.Series(label_scores)
  ax = s.plot.kde()


def report_binary_metrics(predictions, prediction_scores, label):
  binary_predictions = []
  binary_labels = []
  binary_prediction_scores = []
  binary_original_index = []
  for i in range(len(predictions)):
    prediction = predictions[i]
    if(prediction != 2):
      binary_predictions.append(prediction)
      binary_labels.append(label[i])
      binary_prediction_scores.append(prediction_scores[i])
      binary_original_index.append(i)
  return binary_predictions, binary_labels, binary_prediction_scores, binary_original_index


def report(binary_labels, binary_predictions):
	print(f'Accuracy = {metrics.accuracy_score(binary_labels, binary_predictions)}')
	print(f'F1 =  {metrics.f1_score(binary_labels, binary_predictions)}')
	print(f'Precision =  {metrics.precision_score(binary_labels, binary_predictions)}')
	print(f'Recall =  {metrics.recall_score(binary_labels, binary_predictions)}')



def to_numeric_labels(non_num_lable):
	labels = []
	for label in non_num_lable:
		if(label == 'notsarc'):
			labels.append(0)
		else:
			labels.append(1)
	return labels



