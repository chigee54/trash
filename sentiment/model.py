
import os, sys

import sklearn
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import tensorflow_datasets
from transformers import *
import torch


class FeatureBuilder():
	def __init_(self, method='tfidf'):
		self.method = method

	def get_feature(train_data, test_data, tokenizer=None):
		if self.method=='tfidf':
			return get_tfidf_feature(train_data, test_data)
		elif self.method=='sentence piece':
			return get_bert_feature(train_data, test_data)

	def get_tfidf_feature(train_data, test_data):
		X_train_data, y_train = zip(*train_data)
		X_test_data, y_test = zip(*train_data)

		vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()  # 定义一个tf-idf的vectorizer
		X_train = vectorizer.fit_transform(X_train_data) # 训练数据的特征
		X_test = vectorizer.transform(X_test_data) # 测试数据的特征
		return X_train, y_train, X_test, y_test

	def get_bert_feature(train_data, test_data, tokenizer):
		return tokenizer.encode(train_data), tokenizer.encode(test_data)



class LinearModel():
	def __init__(self):
		self.algorithm = 'LR'
		grid={"C":numpy.logspace(-3,3,7)}
		self.logreg=LogisticRegression(solver='lbfgs',max_iter=1000)
		self.logreg_cv=sklearn.model_selection.GridSearchCV(logreg,grid,cv=10,scoring='f1')

	def train(self, X_train, y_train):
		self.logreg_cv.fit(X_train,y_train)
		print(sklearn.metrics.classification_report(y_test, y_pred))

	def predict(self, X_test):
		y_pred = logreg_cv.predict(X_test)



# TODO ...
class NNModel():
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
		self.model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
		self.init_model()

	def init_model(self):
		optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
		model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

	def get_tokenizer(self):
		return self.tokenizer

	def train(self, X_train, y_train):

		input_ids = torch.tensor(X_train)
		history = self.model.fit(input_ids, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)
		self.model.save_pretrained('./save/')

	def predict(self, X_test):
		
		return self.model(torch.tensor(X_test)).argmax().item()

