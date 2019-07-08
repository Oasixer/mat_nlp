## Includes Code from https://github.com/AiswaryaSrinivas/DataScienceWithPython/blob/master/CRF%20POS%20Tagging.ipynb

import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
import sys
import os.path
import pickle
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF, metrics, scorers
from collections import Counter


class PosTagger:
    def __init__(self):
        self.importCorpus()
        self.splitTrainingData()
        self.train()

    def importCorpus(self):
        self.tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
        print("Number of Tagged Sentences ", len(self.tagged_sentence))
        self.tagged_words=[tup for sent in self.tagged_sentence for tup in sent]
        print("Total number of tagged Words ", len(self.tagged_words))
        self.vocab=set([word for word,tag in self.tagged_words])
        print("Vocabulary of the Corpus ", len(self.vocab))
        self.tags=set([tag for word,tag in self.tagged_words])
        print("number of tags in the corpus ", len(self.tags))

    def splitTrainingData(self):
        self.train_set, self.test_set = train_test_split(self.tagged_sentence, test_size=0.2, random_state=1234)
        print("Number of Sentences in Training Data ", len(self.train_set))
        print("number of Sentences in Testing Data ", len(self.test_set))

    def train(self):
        self.x_train, self.y_train=PosTagger.prepareData(self.train_set)
        self.x_test, self.y_test=PosTagger.prepareData(self.test_set)

        # CRF with default paramaters
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.01,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self.crf.fit(self.x_train, self.y_train)

        print("Training complete!")

    def predictionStats(self):
        self.y_pred = self.crf.predict(self.x_test)
        print("F1 Score test: ", metrics.flat_f1_score(self.y_test, self.y_pred, average='weighted', labels=self.crf.classes_))
        self.y_pred_train = self.crf.predict(self.x_train)
        print("F1 Score train: ", metrics.flat_f1_score(self.y_train, self.y_pred_train,average='weighted',labels=self.crf.classes_))
        print("Accuracy Test: ", metrics.flat_accuracy_score(self.y_test, self.y_pred))
        print("Accuracy Train: ", metrics.flat_accuracy_score(self.y_train, self.y_pred_train))
        print(metrics.flat_classification_report(
            self.y_test, self.y_pred, labels=self.crf.classes_, digits=3
        ))
        print("Number of Transition Features ", len(self.crf.transition_features_))
        print(Counter(self.crf.transition_features_).most_common(20))
        print(Counter(self.crf.transition_features_).most_common()[-20:])
        print("Number of State Features ", len(self.crf.state_features_))
        print(Counter(self.crf.state_features_).most_common(20))
        print(Counter(self.crf.state_features_).most_common()[-20:])

    @staticmethod
    def features(sentence, index):
        ### Sentece is of the form [w1, w2, w3, ...], index is the possition of the word in the sentence

        return {
            'is_first_capitol': int(sentence[index][0].isupper()),
            'is_first_word': int(index==0),
            'is_last_word': int(index==len(sentence)-1),
            'is_complete_capital': int(sentence[index].upper() == sentence[index]),
            'prev_word' : '' if index == 0 else sentence[index-1],
            'next_word' : '' if index==len(sentence)-1 else sentence[index+1],
            'is_numeric': int(sentence[index].isdigit()),
            'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index])))),
            'prefix_1': sentence[index][0],
            'prefix_2': sentence[index][:2],
            'prefix_3': sentence[index][:3],
            'prefix_4': sentence[index][:4],
            'sufix_1': sentence[index][-1],
            'sufix_2': sentence[index][-2:],
            'sufix_3': sentence[index][-3:],
            'sufix_4': sentence[index][-4:],
            'word_has_hyphen': 1 if '-' in sentence[index] else 0
        }

    @staticmethod
    def untag(sentence):
        return [word for word, tag in sentence]

    @staticmethod
    def prepareData(tagged_sentences):
        x,y=[],[]
        for sentence in tagged_sentences:
            x.append([PosTagger.features(PosTagger.untag(sentence), index) for index in range(len(sentence))])
            y.append([tag for word, tag in sentence])

        return x,y

def saveModel(model):
    with open("crfModel.pkl", "wb") as modelBin:
        pickle.dump(model, modelBin)

def loadModel():
    if os.path.exists("crfModel.pkl"):
        with open("crfModel.pkl", "rb") as modelBin:
            return pickle.load(modelBin)
    else:
        raise Exception("The CRF Model appears to be missing. Please create one by running \n\tpython CRF_model init \nin your working directory.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            """Invalid argument. Valid arguments are:
                \t 1. init (initializes crf, first time use)
                \t 2. train
                \t 3. updateCorpus""")

    if sys.argv[1] is "init":
        model = PosTagger()
        saveModel(model)

    if sys.argv[1] is "train":
        model = loadModel()
        model.train()
        saveModel(model)

    if sys.argv[1] is "updateCorpus":
        model = loadModel()
        model.importCorpus()
        saveModel(model)


## Import the corpus
# tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
# print("Number of Tagged Sentences ", len(tagged_sentence))
# tagged_words=[tup for sent in tagged_sentence for tup in sent]
# print("Total number of tagged Words ", len(tagged_words))
# vocab=set([word for word,tag in tagged_words])
# print("Vocabulary of the Corpus ", len(vocab))
# tags=set([tag for word,tag in tagged_words])
# print("number of tags in the corpus ", len(tags))

## spLit the corpus into training and test and data for our model
# train_set, test_set = train_test_split(tagged_sentence, test_size=0.2, random_state=1234)
# print("Number of Sentences in Training Data ", len(train_set))
# print("number of Sentences in Testing Data ", len(test_set))


# x_train, y_train=prepareData(train_set)
# x_test, y_test=prepareData(test_set)

# # CRF with default paramaters
# crf = CRF(
#     algorithm='lbfgs',
#     c1=0.01,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True
# )

# crf.fit(x_train, y_train)

# y_pred=crf.predict(x_test)

# print("F1 Score test: ", metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=crf.classes_))


# y_pred_train=crf.predict(x_train)
# print("F1 Score train: ", metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=crf.classes_))

# print("Accuracy Test: ", metrics.flat_accuracy_score(y_test,y_pred))

# print("Accuracy Train: ", metrics.flat_accuracy_score(y_train,y_pred_train))

# print(metrics.flat_classification_report(
#     y_test, y_pred, labels=crf.classes_, digits=3
# ))

# print("Number of Transition Features ", len(crf.transition_features_))


# print(Counter(crf.transition_features_).most_common(20))

# print(Counter(crf.transition_features_).most_common()[-20:])

# print("Number of State Features ", len(crf.state_features_))

# print(Counter(crf.state_features_).most_common(20))

# print(Counter(crf.state_features_).most_common()[-20:])



