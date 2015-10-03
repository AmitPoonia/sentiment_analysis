#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import re
import glob
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.externals import joblib
from matplotlib import pylab  as pl
import matplotlib.pyplot as plt


def wordlist(raw_text, remove_stopwords=False):
	"""
	Removes non-letters, and stop-words(optionally)
	"""
	clean_text = re.sub("[^a-zA-Z]", " ", raw_text)
	words = clean_text.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [word for word in words if word not in stops]
	return words


def preprocess(texts):
	"""
	Basic pre-processing of text data
	"""
	list_of_lists = map(wordlist, texts)
	return list_of_lists


def remove_tuples(tuples_list):
	"""
	Removes the tuple structure and returns only 2nd tuple items as a list
	"""
	tupleless = [tup[1] for tup in tuples_list]
	return tupleless


def get_data(directory_path, file_extension='txt'):
	"""
	Fetches the data from multiple files and returns them as a dictionary
	where every key, value pair is: the name of file, and its content.
	"""
	data_dict = dict()
	files = glob.glob(directory_path+'*.'+file_extension)
	for file_name in files:
		data_dict[file_name] = open(file_name, 'r').read()
	return data_dict


def get_best(X_train, y_train, X_test, y_test):
	"""
	Searches for best parameters, and in this case only one parameter:
	The no. of topics in LSA/LSI based dimension reduction of text data.
	The best no. of topics decided by ROC area under curve.
	"""
	max_roc_auc = 0.0
	best_num_topics = None
	X_train_ = None
	X_test_ = None

	num_topics_set = [num*10 for num in range(1,11)]
	for num_topics in num_topics_set:
		X_train_, X_test_ = vectorize(X_train, X_test, num_topics)
		model = train(X_train_, y_train)
		_, _, _, roc_auc = test(model, X_test_, y_test)
		print(' --- ', roc_auc, num_topics)
		if roc_auc >= max_roc_auc:
			max_roc_auc = roc_auc
			best_num_topics = num_topics

	return best_num_topics, max_roc_auc


def vectorize(X_train, X_test, num_topics, load_dict=None, load_lsi=None):
	"""
	Vectorizes the text data first by converting it into a bag-of-words
	and then using a LSI(Latent Semantic Indexing) model to represent that
	data into low dimensional vector space.
	"""
	text_train = preprocess(X_train)
	test_train = preprocess(X_test)
	if load_dict is None:
		dictionary = corpora.Dictionary(text_train+test_train)
		dictionary.save('sentiment.dict')
	else:
		dictionary = corpora.Dictionary.load(load_dict)

	corpus_train = map(dictionary.doc2bow, text_train)
	corpus_test = map(dictionary.doc2bow, text_train)

	if load_lsi is None:
		lsi = models.LsiModel(corpus_train+corpus_test, id2word=dictionary,
													num_topics=num_topics)
		lsi.save('sentiment.lsi')
	else:
		lsi = models.LsiModel.load(load_lsi)

	X_train = np.array(map(remove_tuples, lsi[corpus_train]))   
	X_test = np.array(map(remove_tuples, lsi[corpus_test]))

	return X_train, X_test

	
def train(X_train, y_train, load_model=None, C=1.0):
	"""
	Trains a classification model with mostly default settings, or
	loads an existing model.
	"""
	if load_model is None:
		svc = SVC(C=C, probability=True)
		model = svc.fit(X_train, y_train)
		_ = joblib.dump(model, 'sentiment.svc', compress=9)
	else:
		model = joblib.load(load_model)

	return model


def test(model, X_test, y_test):
	"""
	Tests the trained classifier, and returns different evaluation metrics
	"""

	pred = model.predict(X_test)
	cr = classification_report(y_test, pred,
								target_names=['1','0'])
	pred_probas = model.predict_proba(X_test)[:,1]
	fpr, tpr, _ = roc_curve(y_test, pred_probas)
	roc_auc = auc(fpr, tpr)

	return cr, fpr, tpr, roc_auc


def show_test(cr, fpr, tpr, roc_auc, model):
	"""
	Prints the results of tests, and plots the ROC curve
	"""
	print("Classification report on test set for classifier:")
	print(model)
	print()
	print(cr)

	plt.plot(fpr,tpr,label='area under curve = %.2f' %roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc='lower right')

	plt.show()


def get_sentiment(text, load_model='sentiment.svc', 
		load_dict='sentiment.dict', load_lsi='sentiment.lsi'):
	"""
	Returns the sentiment of some random text snippet
	"""
	text = wordlist(text)
	dictionary = corpora.Dictionary.load(load_dict)
	text_bow = dictionary.doc2bow(text)
	lsi = models.LsiModel.load(load_lsi)
	text_vector = remove_tuples(lsi[text_bow])
	model = joblib.load(load_model)
	sentiment  = model.predict(text_vector)

	return sentiment


def main():

	# hyper-parameters
	num_topics = 25
	parameter_search = False

	# getting data from movie reviews datasets
	X_train_pos = get_data('datasets/train/pos/').values()
	X_train_neg = get_data('datasets/train/neg/').values()
	X_test_pos = get_data('datasets/test/pos/').values()
	X_test_neg = get_data('datasets/test/neg/').values()

	# initializes numpy arrays, zero for negative and one for positive
	y_train_pos = np.ones(len(X_train_pos))
	y_train_neg = np.zeros(len(X_train_neg))
	y_test_pos = np.ones(len(X_test_pos))
	y_train_neg = np.zeros(len(X_test_neg))

	X_train = X_train_pos + X_train_neg
	y_train = np.concatenate((y_train_pos, y_train_neg))
	X_test = X_test_pos + X_test_neg
	y_test = np.concatenate((y_test_pos, y_train_neg))

	# If search flag is on, it returns most optimal 'no. of topics' for
	# LSA model, and corresponding ROC area under curve value.
	if parameter_search:

		num_topics, roc_auc = get_best(X_train, y_train, X_test, y_test)
		print(num_topics)
		print(roc_auc)

	else:

		# Vectorizes data using Latent Semantic Analysis(LSA)
		X_train, X_test = vectorize(X_train, X_test, num_topics, 
							'sentiment.dict', 'sentiment.lsi')
		# train a binary classification model
		model = train(X_train, y_train, 'sentiment.svc')
		# Tests the model and prints different metrics
		cr, fpr, tpr, roc_auc = test(model, X_test, y_test)
		# Prints test results
		show_test(cr, fpr, tpr, roc_auc, model)

		# Tests some random text for sentiment
		text_sample = "I didn't like that movie!"
		sentiment = get_sentiment(text_sample)
		print(sentiment)



if __name__ == '__main__':
	main()
