#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import re
from gensim import corpora, models
from collections import defaultdict
import glob
import numpy as np
from nltk.corpus import stopwords
import operator

from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from matplotlib import pylab  as pl
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets.base import Bunch
from sklearn.svm import SVC, LinearSVC



def wordlist(body, remove_stopwords=False):
	""" convert a document to a sequence of words, optionally removing stop words.  Returns a list of words."""

	# Remove non-letters
	text = re.sub("[^a-zA-Z]", " ", body)

	# convert words to lower case and split them
	words = text.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]

	return words



def clean_body(texts):
	list_of_lists = []
	for text in texts:
		list_of_lists.append(wordlist(text, remove_stopwords=False))
	return list_of_lists

def remove_tuples(ob):
	oob = []
	for o in ob:
		oob.append(o[1])
	return oob

def get_data(directory_path):
	data_dict = dict()
	files = glob.glob(directory_path+'*.txt')
	for file_name in files[0:5000]:
		fin = open(file_name, 'r').read()
		title = file_name.split('/')[-1].split('.')[0]
		data_dict[title] = fin

	return data_dict



def main():

	# hyper-parameters

	num_topics = 10


	# getting data from movie datasets

	X_train_pos = get_data('datasets/train/pos/').values()
	X_train_neg = get_data('datasets/train/neg/').values()
	X_test_pos = get_data('datasets/test/pos/').values()
	X_test_neg = get_data('datasets/test/neg/').values()

	y_train_pos = np.ones(len(X_train_pos))
	y_train_neg = np.zeros(len(X_train_neg))
	y_test_pos = np.ones(len(X_test_pos))
	y_train_neg = np.zeros(len(X_test_neg))

	X_train = X_train_pos + X_train_neg
	y_train = np.concatenate((y_train_pos,y_train_neg))
	X_test = X_test_pos + X_test_neg
	y_test = np.concatenate((y_test_pos,y_train_neg))

	##print(X_test[-1])
	# vectorize data

	text_train = clean_body(X_train)
	test_train = clean_body(X_test)
	dictionary = corpora.Dictionary(text_train+test_train)
	corpus_train = map(dictionary.doc2bow, text_train)
	corpus_test = map(dictionary.doc2bow, text_train)

	lsi = models.LsiModel(corpus_train+corpus_test, id2word=dictionary, num_topics=num_topics)

	lsi_vec_train = lsi[corpus_train]
	lsi_vec_test = lsi[corpus_test]

	# converting vectors from tuples to lists
	# TODO: conver below code into mapping
	X_train_ = []
	X_test_ = []

	for ob in lsi_vec_train:
		X_train_.append(remove_tuples(ob))
	for ob in lsi_vec_test:
		X_test_.append(remove_tuples(ob))

	X_train = np.zeros((len(y_train), num_topics))
	X_test = np.zeros((len(y_test), num_topics))

	for i, v in enumerate(X_train_):
		for j, u in enumerate(v):
			X_train[i][j] = u

	for i, v in enumerate(X_test_):
		for j, u in enumerate(v):
			X_test[i][j] = u


	svc = SVC(probability=True)
	model = svc.fit(X_train, y_train)
	pred = model.predict(X_test)

	print("Classification report on test set for classifier:")
	print(model)
	print()
	print(classification_report(y_test, pred,
								target_names=['1','0']))

	cm = confusion_matrix(y_test, pred)
	print("Confusion matrix:")
	print(cm)

	# Show confusion matrix
	pl.matshow(cm)
	pl.title('Confusion matrix of the classifier')
	pl.colorbar()

	print(model.predict_proba(X_test[-1]))
	print(model.predict(X_test[-1]))






if __name__ == '__main__':
	main()