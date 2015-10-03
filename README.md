
===================================================
Binary Sentiment Classification of Movie Reviews
===================================================

The approach to do sentiment analysis here has two parts:
	a) Data Representation:- Pre-processing and transforming the input text such that it can
		be given to a classifier to train and test, here using LSA/LSI to
		transform the text. Latent Semantic Analysis creates low dimensional
		space for text data while keeping only n-best features, and these
		feature values can be used to find similarity among different text
		documents. 
	b) Data Classification:- Selecting a classifier and training+testing it on transformed
		text data. Using a Support Vector Machines here, based on libsvm
		which works very well for a simple binary classification on low no. of 
		samples, as in this case.


The Work-flow:
0. Loads the movie reviews text files, train/test split already done
1. Does some text pre-processing
2. Creates a dictionary and subsequent bag-of-words corpus from text data
3. Fits an LSI model with the corpus 
4. Transform the data in low dimensional LSI vector space
5. Trains a SVM for binary classification(Positive/Negative Sentiment)
6. Tests the classifier with test data
7. Plots ROC curve and prints evaluation metrics


One can also do greedy search for hyper-parameters by setting 'parameter_search'
flag 'True'. In current setting, only one parameter: number of topics
for LSI model, is being searched.

In the default settings, the program loads already trained models and gives
the test results, to train new models one can just not load the pre-existing models either by
setting them as 'None' or just remove the arguments from function call altogether and program
will proceed with training new models.



References:

https://radimrehurek.com/gensim/tutorial.html
http://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis