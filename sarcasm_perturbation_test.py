from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
import random
import evaluate

def test(sarc_percentage):
	# BERT-base and RoBERTa-base models fine-tuned on sentiment anlaysis datasets (SST-2 and IMDB). 
	models = [
	          'textattack/bert-base-uncased-SST-2',
	          'textattack/roberta-base-SST-2',
	          'textattack/bert-base-uncased-imdb',
	          'textattack/roberta-base-imdb',
	]
	
	# Datasets - GEN, HYP, RQ, SemEval (sarcasm only) separatly.
	gen_sarc = pd.read_csv('/scratch/ec2684/GEN-sarc-notsarc.csv')
	hyp_sarc = pd.read_csv('/scratch/ec2684/HYP-sarc-notsarc.csv')
	rg_sarc = pd.read_csv('/scratch/ec2684/RQ-sarc-notsarc.csv')
	sem_eval = pd.read_csv('/scratch/ec2684/SemEval2018-T3-train-taskA.csv')

	# A csv file that contains the model prediction results of testing the above four sarcastic models.
	sarc_model_pred_report = pd.read_csv('/scratch/ec2684/report.csv')

	## Generating a dataset with only sarcastic examples from the above four datasets.
	gen_sarc_data = evaluate.extrac_sarc_only(gen_sarc, False)
	hyp_sarc_data = evaluate.extrac_sarc_only(hyp_sarc, False)
	rg_sarc_data = evaluate.extrac_sarc_only(rg_sarc, False)
	sem_sarc_data = evaluate.extrac_sarc_only(sem_eval, True)

	dataset_name = ['GEN-sarc-notsarc.csv','HYP-sarc-notsarc.csv','RQ-sarc-notsarc.csv','SemEval2018-T3-train-taskA.csv']
	sarc_datasets = [gen_sarc_data,hyp_sarc_data,rg_sarc_data,sem_sarc_data]
	sarc_labels = []

	# Labels for the sarcastic examples only datasets.
	for dataset in sarc_datasets:
		sarc_labels.append(np.zeros(len(dataset), dtype ='int'))

	## IMDB dataset instantiation.
	imdb_test = load_dataset('imdb', split = 'test')
	imdb_test_positive = imdb_test.filter(lambda example: example['label'] == 1)
	imdb_test_negative = imdb_test.filter(lambda example: example['label'] == 0)
	imdb_test_positive = imdb_test_positive.sort(column = 'text')
	imdb_test_negative = imdb_test_negative.sort(column = 'text')
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	imdb_test_positive = imdb_test_positive.filter(lambda example: len(tokenizer(example['text'])['attention_mask']) < 512)
	imdb_test_negative = imdb_test_negative.filter(lambda example: len(tokenizer(example['text'])['attention_mask']) < 512)



	# Extracting the samples from the sarcastic dataset that the models have predicted as being negative.
	# Dictionary of such data
	neg_pred_sarc_dataset = {}

	for i in range(6):
	    k = len(dataset_name)*i
	    dict_temp = {}
	    for j in range(len(dataset_name)):
	        labels = sarc_model_pred_report['Correct index'][k + j]
	        neg_pred_index = list(map(int, labels[1:-1].split(',')))
	        truncated = []
	        for index in neg_pred_index:
	            truncated.append(sarc_datasets[j][index])
	        dict_temp[sarc_model_pred_report['Dataset'][k + j]] = truncated

	    neg_pred_sarc_dataset[sarc_model_pred_report['Model'][k]] = dict_temp


	comprehensive_results = []
	
	index = 1
	clean_comprehensive_results = []
	perturbation_first_results = []
	perturbation_last_results = []

	var = 50
	imdb_token_length = 400
	pert_token_length = 100
	imdb_test_positive = imdb_test_positive.filter(lambda example: len(tokenizer(example['text'])['attention_mask']) < imdb_token_length+var)
	imdb_test_positive = imdb_test_positive.filter(lambda example: imdb_token_length-var < len(tokenizer(example['text'])['attention_mask']))
	imdb_test_negative = imdb_test_negative.filter(lambda example: len(tokenizer(example['text'])['attention_mask']) < imdb_token_length+var)
	imdb_test_negative = imdb_test_negative.filter(lambda example: imdb_token_length-var < len(tokenizer(example['text'])['attention_mask']))

	print(f'{sarc_percentage} keeping {imdb_token_length}/{pert_token_length}')
	# Iterating through the models and evaluating the results.
	for model in models:
		# Initialzing the model.
		tokenizer = AutoTokenizer.from_pretrained(model)
		model_init = AutoModelForSequenceClassification.from_pretrained(model)
		nlp_pipeline = pipeline("sentiment-analysis", model=model_init, tokenizer=tokenizer, framework="pt", device=0)

		print(f'\n\n({index}) Report for {model}.')
		subindex = 0
		for dataset in sarc_datasets:
			print(f'''\n({index}-{subindex+1}) Testing on {dataset_name[subindex]} dataset on only sarcastic data.\n''')
			passage_raw = neg_pred_sarc_dataset[model][dataset_name[subindex]]
			output = evaluate.length_threshold(passage_raw, pert_token_length-var, pert_token_length+var, tokenizer)
			print(f'''\n{np.around(output['proportion'],4)*100}% of {dataset_name[subindex]} sarcastic data passed threshold test.\n''')
			
			# Shuffling the order of random (sarcastic) perturbations.
			passage = output['dataset']
			random.shuffle(passage)
			imdb_length = len(imdb_test_positive['text'])
			
			print(f'\nTesting IMDB positive dataset:\n IMDB len: {imdb_length}\n Passage len: {len(passage)}')
			n = imdb_length
			imdb_passage = imdb_test_positive['text']
			label = np.ones(n, dtype = int)


			print(f'\nWithout perturbations:')
			predictions, prediction_scores = evaluate.evaluate(imdb_passage, nlp_pipeline)
			binary_predictions_no_perturb, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, label)
			acc = evaluate.report_acc(binary_labels, binary_predictions_no_perturb)
			clean_summary = evaluate.summary(label, predictions, prediction_scores, model, dataset_name[subindex], acc)
			
			print(f'\nWith perturbations: perturbation + IMDB order.')
			perturbed_passage, ratio = evaluate.merge(imdb_passage, passage, False)
			predictions, prediction_scores = evaluate.evaluate(perturbed_passage, nlp_pipeline)
			binary_predictions_perturb, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, label)
			acc = evaluate.report_acc(binary_labels, binary_predictions_perturb)
			no_change = np.count_nonzero(np.asarray(binary_predictions_no_perturb) == np.asarray(binary_predictions_perturb))
			differences = len(binary_predictions_no_perturb) - no_change
			print(f'''\nFrom {len(binary_predictions_no_perturb)} predictions, perturbation + IMDB changed:\n {differences} labels\n {no_change} labels remained the same''')
			perturb_first_summary = evaluate.perturb_summary(label, predictions, prediction_scores, model, dataset_name[subindex], acc, ratio, differences, no_change)


			print(f'\nWith perturbations: IMDB + perturbation order.')
			perturbed_passage, ratio = evaluate.merge(imdb_passage, passage, True)
			predictions, prediction_scores = evaluate.evaluate(perturbed_passage, nlp_pipeline)
			binary_predictions_perturb, binary_labels, binary_prediction_scores, binary_original_index = evaluate.report_binary_metrics(predictions, prediction_scores, label)
			acc = evaluate.report_acc(binary_labels, binary_predictions_perturb)
			no_change = np.count_nonzero(np.asarray(binary_predictions_no_perturb) == np.asarray(binary_predictions_perturb))
			differences = len(binary_predictions_no_perturb) - no_change
			print(f'''\nFrom {len(binary_predictions_no_perturb)} predictions, IMDB + perturbation changed:\n {differences} labels\n {no_change} labels remained the same''')
			perturb_last_summary = evaluate.perturb_summary(label, predictions, prediction_scores, model, dataset_name[subindex], acc, ratio, differences, no_change)

			clean_comprehensive_results.append(clean_summary)
			perturbation_first_results.append(perturb_first_summary)
			perturbation_last_results.append(perturb_last_summary)
			subindex = subindex+1
		index = index+1
	return clean_comprehensive_results,perturbation_first_results,perturbation_last_results