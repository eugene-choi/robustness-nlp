import numpy as np
import torch
import pandas as pd

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn import metrics
import evaluate
import sys


# sem_eval = pd.read_csv('/scratch/ec2684/SemEval2018-T3-train-taskA.csv')
# passage = sem_eval['Tweet text']
# label = sem_eval['Label']

gen_sarc = pd.read_csv('/scratch/ec2684/GEN-sarc-notsarc.csv')
hyp_sarc = pd.read_csv('/scratch/ec2684/HYP-sarc-notsarc.csv')
rg_sarc = pd.read_csv('/scratch/ec2684/RQ-sarc-notsarc.csv')

gen_passage = gen_sarc['text']
gen_label = evaluate.to_numeric_labels(gen_sarc['class'])

hyp_passage = hyp_sarc['text']
hyp_label = evaluate.to_numeric_labels(hyp_sarc['class'])

rg_passage = rg_sarc['text']
rg_label = evaluate.to_numeric_labels(rg_sarc['class'])

#passage = gen_passage + hyp_passage + rg_passage
#label = gen_label + hyp_label + rg_label

print('(1) BERT expriment using:')
print('typeform/distilbert-base-uncased-mnli')
# BERT
bert_model_mnli = 'typeform/distilbert-base-uncased-mnli'
tokenizer_bert_mnli = AutoTokenizer.from_pretrained(bert_model_mnli)
model_bert_mnli = AutoModelForSequenceClassification.from_pretrained(bert_model_mnli)
pipe_bert = pipeline("sentiment-analysis", model=model_bert_mnli, tokenizer=tokenizer_bert_mnli, framework="pt", device=0)
print('GEN-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(gen_passage, pipe_bert)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, gen_label)
evaluate.report(binary_labels, binary_predictions)
print('HYP-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(hyp_passage, pipe_bert)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, hyp_label)
evaluate.report(binary_labels, binary_predictions)
print('RQ-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(rg_passage, pipe_bert)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, rg_label)
evaluate.report(binary_labels, binary_predictions)


print('\n\n(2) RoBERTa expriment using:')
print('roberta-large-mnli')
# RoBERTa
rob_large = 'roberta-large-mnli'
rob_tokenizer = AutoTokenizer.from_pretrained(rob_large)
rob_model = AutoModelForSequenceClassification.from_pretrained(rob_large)
pipe_rob = pipeline("sentiment-analysis", model=rob_model, tokenizer=rob_tokenizer, framework="pt", device=0)
print('GEN-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(gen_passage, pipe_rob)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, gen_label)
evaluate.report(binary_labels, binary_predictions)
print('HYP-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(hyp_passage, pipe_rob)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, hyp_label)
evaluate.report(binary_labels, binary_predictions)
print('RQ-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(rg_passage, pipe_rob)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, rg_label)
evaluate.report(binary_labels, binary_predictions)



print('\n\n(3) DeBERTa expriment using:')
print('microsoft/deberta-base-mnli')
# DeBERTa
#deberta_name = 'microsoft/deberta-xlarge-mnli'
deberta_name = 'microsoft/deberta-base-mnli'
tokenizer_deb = AutoTokenizer.from_pretrained(deberta_name)
model_deb = AutoModelForSequenceClassification.from_pretrained(deberta_name)
pipe_deb = pipeline('sentiment-analysis', model=model_deb, tokenizer = tokenizer_deb, framework = 'pt', device = 0)
print('GEN-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(gen_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, gen_label)
evaluate.report(binary_labels, binary_predictions)
print('HYP-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(hyp_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, hyp_label)
evaluate.report(binary_labels, binary_predictions)
print('RQ-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(rg_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, rg_label)
evaluate.report(binary_labels, binary_predictions)


print('\n\n(4) DeBERTa expriment using:')
print('microsoft/deberta-large-mnli')
# DeBERTa
#deberta_name = 'microsoft/deberta-xlarge-mnli'
deberta_name = 'microsoft/deberta-large-mnli'
tokenizer_deb = AutoTokenizer.from_pretrained(deberta_name)
model_deb = AutoModelForSequenceClassification.from_pretrained(deberta_name)
pipe_deb = pipeline('sentiment-analysis', model=model_deb, tokenizer = tokenizer_deb, framework = 'pt', device = 0)
print('GEN-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(gen_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, gen_label)
evaluate.report(binary_labels, binary_predictions)
print('HYP-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(hyp_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, hyp_label)
evaluate.report(binary_labels, binary_predictions)
print('RQ-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(rg_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, rg_label)
evaluate.report(binary_labels, binary_predictions)


print('\n\n(5) DeBERTa expriment using:')
print('microsoft/deberta-xlarge-mnli')
# DeBERTa
deberta_name = 'microsoft/deberta-xlarge-mnli'
tokenizer_deb = AutoTokenizer.from_pretrained(deberta_name)
model_deb = AutoModelForSequenceClassification.from_pretrained(deberta_name)
pipe_deb = pipeline('sentiment-analysis', model=model_deb, tokenizer = tokenizer_deb, framework = 'pt', device = 0)
print('GEN-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(gen_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, gen_label)
evaluate.report(binary_labels, binary_predictions)
print('HYP-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(hyp_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, hyp_label)
evaluate.report(binary_labels, binary_predictions)
print('RQ-sarc-notsarc.csv')
predictions, prediction_scores = evaluate.evaluate(rg_passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, rg_label)
evaluate.report(binary_labels, binary_predictions)


"""
print('\n\n(6) DeBERTa expriment using:')
print('microsoft/deberta-v2-xlarge-mnli')
# DeBERTa
deberta_name = 'microsoft/deberta-v2-xlarge-mnli'
tokenizer_deb = AutoTokenizer.from_pretrained(deberta_name)
model_deb = AutoModelForSequenceClassification.from_pretrained(deberta_name)
pipe_deb = pipeline('sentiment-analysis', model=model_deb, tokenizer = tokenizer_deb, framework = 'pt', device = 0)
predictions, prediction_scores = evaluate.evaluate(passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, label)
evaluate.report(binary_labels, binary_predictions)


print('\n\n(7) DeBERTa expriment using:')
print('microsoft/deberta-v2-xxlarge-mnli')
# DeBERTa
deberta_name = 'microsoft/deberta-v2-xxlarge-mnli'
tokenizer_deb = AutoTokenizer.from_pretrained(deberta_name)
model_deb = AutoModelForSequenceClassification.from_pretrained(deberta_name)
pipe_deb = pipeline('sentiment-analysis', model=model_deb, tokenizer = tokenizer_deb, framework = 'pt', device = 0)
predictions, prediction_scores = evaluate.evaluate(passage, pipe_deb)
binary_predictions, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, label)
evaluate.report(binary_labels, binary_predictions)

sys.stdout.close()
"""



