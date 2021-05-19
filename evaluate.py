import numpy as np
import torch
import pandas as pd
import csv
import re
import random
from sklearn import metrics


# Function that standardizes the prediction labels of different models by converting them into 0/1.
def convert_to_numeric_label(predictions):
  if(predictions == 'ENTAILMENT'):
    return 0
  elif(predictions == 'CONTRADICTION'):
    return 1
  elif(predictions == 'NEUTRAL'):
    return 2
  elif(predictions == 'LABEL_1'):
    return 1
  elif(predictions == 'LABEL_0'):
    return 0
  elif(predictions == 'POSITIVE'):
    return 1
  elif(predictions == 'NEGATIVE'):
    return 0

# Function that evaluates the given passage (dataset) using the huggingface pipeline object given by the param.
def evaluate(passage, pipeline_obj):
  predictions = []
  predictions_scores = []
  for text in passage:
    # Models automatically truncates the input text if it has token length > 512.
    prediction = pipeline_obj(text, truncation = True)[0]
    predictions.append(convert_to_numeric_label(prediction['label']))
    predictions_scores.append(prediction['score'])
  return predictions, predictions_scores

# A helper function that converts 
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

# Function for reporting result metrics.
def report(binary_labels, binary_predictions):
	print(f'Accuracy = {metrics.accuracy_score(binary_labels, binary_predictions)}')
	print(f'F1 =  {metrics.f1_score(binary_labels, binary_predictions)}')
	print(f'Precision =  {metrics.precision_score(binary_labels, binary_predictions)}')
	print(f'Recall =  {metrics.recall_score(binary_labels, binary_predictions)}')

# Function for reporting result accuracy only.
def report_acc(binary_labels, binary_predictions):
  acc = metrics.accuracy_score(binary_labels, binary_predictions)
  print(f'Accuracy = {acc}')
  return acc

# A label preprocessing function for sarcasm dataset.
def to_numeric_labels(non_num_lable):
	labels = []
	for label in non_num_lable:
		if(label == 'notsarc'):
			labels.append(1)
		else:
			labels.append(0)
	return labels

def extrac_sarc_only(dataset, is_sem_eval):
  if(is_sem_eval):
    passage = dataset['Tweet text']
    label = dataset['Label']
    for i in range(len(passage)):
      passage[i] = re.sub(r'http\S+', '', passage[i])
  else:
    passage = dataset['text']
    label = to_numeric_labels(dataset['class'])
  sarc_only= []
  for i in np.argwhere(0 == np.asarray(label)).ravel():
    sarc_only.append(passage[i])
  return sarc_only

def confidence_analysis(label, predictions, prediction_scores):
  correct = []
  wrong = []
  correct_index = []
  for i in range(len(label)):
    if(label[i] == predictions[i]):
      correct.append(prediction_scores[i])
      correct_index.append(i)
    else:
      wrong.append(prediction_scores[i])
  # print(f'\nCorrect prediction confidnce report:\n Mean: {np.mean(correct)}\n Std: {np.std(correct)}')
  # print(f'\nWrong prediction confidnce report:\n Mean: {np.mean(wrong)}\n Std: {np.std(wrong)}')


def summary(label, predictions, prediction_scores, model, dataset_name, acc):
  correct = []
  wrong = []
  correct_index = []
  wrong_index = []
  for i in range(len(label)):
    if(label[i] == predictions[i]):
      correct.append(prediction_scores[i])
      correct_index.append(i)
    else:
      wrong.append(prediction_scores[i])
      wrong_index.append(i)
  # print(f'\nCorrect prediction confidence report:\n Mean: {np.mean(correct)}\n Std: {np.std(correct)}')
  # print(f'\nWrong prediction confidence report:\n Mean: {np.mean(wrong)}\n Std: {np.std(wrong)}')
  return {
    'Model': model,
    'Dataset': dataset_name,
    'Accuracy': acc,
    'Total number of data': len(label),
    'Correct index': correct_index,
    'Wrong index': wrong_index,
    'Correct scores': correct,
    'Correct stats': [np.mean(correct),np.std(correct)],
    'Wrong stats': [np.mean(wrong),np.std(wrong)],
  }

def report_csv(data_dict,title):
    csv_file = f'''{title}.csv'''
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(data_dict[0].keys()))
            writer.writeheader()
            for data in data_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def report_csv(data_dict,title):
    csv_file = f'''{title}.csv'''
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(data_dict[0].keys()))
            writer.writeheader()
            for data in data_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def length_threshold(dataset,a,b,tokenzier):
    n = len(dataset)
    entire_length = [len(tokenizer(data)['attention_mask']) for data in dataset]
    filtered_dataset = []
    np_dataset = np.asarray(entire_length)
    for index in np.argwhere((a<np_dataset)&(np_dataset<b)).ravel():
        filtered_dataset.append(dataset[index])
    return {'dataset':filtered_dataset, 
            'proportion': len(dataset)/n
           }

def merge(dataset, perturbation, end):
  d = len(dataset)
  n = len(perturbation)
  perturbed = []
  ratio = []
  for i in range(d):
    x = dataset[i]
    if(i%n == 0):
      random.shuffle(perturbation)
    y = perturbation[i%n]
    if(end):
      perturbed.append(x + " " + y)
    else:
      perturbed.append(y + " " + x)
    ratio.append(np.around(len(y)/len(x),4))
  return perturbed, ratio


def perturb_summary(label, predictions, prediction_scores, model, dataset_name, acc, ratio, diff, no_change):
  correct = []
  wrong = []
  correct_index = []
  wrong_index = []
  for i in range(len(label)):
    if(label[i] == predictions[i]):
      correct.append(prediction_scores[i])
      correct_index.append(i)
    else:
      wrong.append(prediction_scores[i])
      wrong_index.append(i)
  # print(f'\nCorrect prediction confidence report:\n Mean: {np.mean(correct)}\n Std: {np.std(correct)}')
  # print(f'\nWrong prediction confidence report:\n Mean: {np.mean(wrong)}\n Std: {np.std(wrong)}')
  return {
    'Model': model,
    'Dataset': dataset_name,
    'Accuracy': acc,
    'Total number of data': len(label),
    'Correct index': correct_index,
    'Wrong index': wrong_index,
    'Correct scores': correct,
    'Correct stats': [np.mean(correct),np.std(correct)],
    'Wrong stats': [np.mean(wrong),np.std(wrong)],
    'Ratio': ratio,
    'Differences': diff,
    'No change': no_change,
  }