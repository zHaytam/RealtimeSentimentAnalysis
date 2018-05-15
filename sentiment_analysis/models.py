import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import random
from timeit import default_timer as timer
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, NEGATE
import sys
import string
import re
import gc
import numpy as np


NEGATE.append("n't")
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
reg = re.compile('[%s]' % re.escape(string.punctuation))


def save_text(text, filename):
    file = open(filename, 'w')
    file.write(text)
    file.close()


def extract_words(text, handle_negation=False):
    # words = word_tokenize(text)
    words = []
    for word in word_tokenize(text):
        word = word.lower()
        if word in NEGATE:
            words.append(word)
        elif word in string.punctuation:
            continue
        else:
            nw = reg.sub('', word)
            if len(nw) > 0:
                words.append(nw)

    if handle_negation:
        words = negation_handler(words)

    words = [word for word in words if word not in stopwords]
    words = [stemmer.stem(word) for word in words]
    # words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words


def negation_handler(words):
    pos_tags = [pos_tag[1][:2] for pos_tag in nltk.pos_tag(words)]
    new_words = []
    flag = False

    i = 0
    while i < len(words):
        word = words[i]
        pos_tag = pos_tags[i]

        if word == 'not' or word == "n't" or word == 'no':
            flag = not flag
            i += 1
            continue

        if flag:
            if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB':
                new_words.append('not_' + word)
            if (pos_tag == 'DT' or pos_tag == 'RB') and i + 1 < len(words) and pos_tags[i + 1] == 'JJ':
                if pos_tag == 'RB':
                    new_words.append(word)
                new_words.append('not_' + words[i + 1])
                i += 1
            if pos_tag == 'DT' and i + 1 < len(words) and pos_tags[i + 1] == 'RB' and i + 2 < len(words) and \
                    pos_tags[i + 2] == 'JJ':
                new_words.append(words[i + 1])
                new_words.append('not_' + words[i + 2])
                i += 2

            flag = False
        else:
            new_words.append(word)

        i += 1

    return new_words


def save_model(model, filename):
    file = open(filename, 'wb')
    pickle.dump(model, file)
    file.close()


# Start
start = timer()

positive_file = open('./data/positive_reviews.txt', 'r')
negative_file = open('./data/negative_reviews.txt', 'r')

positive_reviews = [(review, 'pos') for review in positive_file.readlines()]
negative_reviews = [(review, 'neg') for review in negative_file.readlines()]
all_reviews = positive_reviews + negative_reviews
print(len(all_reviews))
# Get all words, bigrams
all_words = []
all_bigrams = []
all_trigrams = []

for review in all_reviews:
    words = extract_words(review[0], handle_negation=True)
    all_words.extend(words)
    bigrams = list(nltk.bigrams(words))
    all_bigrams.extend(['_'.join(bigram) for bigram in bigrams])
    trigrams = list(nltk.trigrams(words))
    all_trigrams.extend(['_'.join(trigram) for trigram in trigrams])

all_words = nltk.FreqDist(all_words)
most_used_words = [w[0] for w in sorted(all_words.items(), key=lambda x: x[1], reverse=True)][:20000]
print('Finished extracting the 20000 most used words.')

all_bigrams = nltk.FreqDist(all_bigrams)
most_used_bigrams = [bg[0] for bg in sorted(all_bigrams.items(), key=lambda x: x[1], reverse=True)][:100]
print('Finished extracting the 100 most used bigrams.')

all_trigrams = nltk.FreqDist(all_trigrams)
most_used_trigrams = [bg[0] for bg in sorted(all_trigrams.items(), key=lambda x: x[1], reverse=True)][:10]
print('Finished extracting the 10 most used trigrams.')


def create_features(text, handle_negation=False):
    words = extract_words(text, handle_negation=handle_negation)
    bigrams = ['_'.join(bigram) for bigram in list(nltk.bigrams(words))]
    trigrams = ['_'.join(trigram) for trigram in list(nltk.trigrams(words))]
    features = {}

    for word in most_used_words:
        features[word] = True if word in words else False
    for bigram in most_used_bigrams:
        features[bigram] = True if bigram in bigrams else False
    for trigram in most_used_trigrams:
        features[trigram] = True if trigram in trigrams else False

    features['compound'] = sia.polarity_scores(text)['compound'] + 1
    return features


data = [(create_features(review, handle_negation=True), cat) for (review, cat) in all_reviews]
# random.shuffle(data)
np.random.shuffle(data)

print('Data shuffled ({} rows).'.format(len(data)))

# training_data = data[:8529]
# testing_data = data[8529:]
# del data
# gc.collect()

# Cross validation
# subset_size = int(len(data) / 10)
# accuracies = []
#
# for i in range(10):
#     test = data[int(i * subset_size):][:subset_size]
#     train = data[:int(i * subset_size)] + data[int((i + 1) * subset_size):]
#     nbc = NaiveBayesClassifier.train(train)
#     acc = accuracy(nbc, test)
#     accuracies.append(acc)
#     print("Naive Bayes Classifier: {:.2f}%".format(acc * 100))
#     gc.collect()
#
# print(sum(accuracies) / len(accuracies))
"""
# Naive Bayes Classifier
nbc = NaiveBayesClassifier.train(data)
print("Naive Bayes Classifier: {:.2f}%".format(accuracy(nbc, data) * 100))
save_model(nbc, './models2.7/nbc.pickle')

# Gaussian Naive Bayes Classifier
# gnbc = SklearnClassifier(GaussianNB())
# gnbc.train(data)
# print("Gaussian Naive Bayes Classifier: {:.2f}%".format(accuracy(gnbc, data) * 100))
# save_model(gnbc, './models2.7/gnbc.pickle')

# Multinomial Naive Bayes Classifier
mnbc = SklearnClassifier(MultinomialNB())
mnbc.train(data)
print("Multinomial Naive Bayes Classifier: {:.2f}%".format(accuracy(mnbc, data) * 100))
save_model(mnbc, './models2.7/mnbc.pickle')

# Bernoulli Naive Bayes Classifier
bnbc = SklearnClassifier(BernoulliNB())
bnbc.train(data)
print("Bernoulli Naive Bayes Classifier: {:.2f}%".format(accuracy(bnbc, data) * 100))
save_model(bnbc, './models2.7/bnbc.pickle')

# Logistic Regression Classifier
lgc = SklearnClassifier(LogisticRegression())
lgc.train(data)
print("Logistic Regression Classifier: {:.2f}%".format(accuracy(lgc, data) * 100))
save_model(lgc, './models2.7/lgc.pickle')

# SGD Classifier
sgdc = SklearnClassifier(SGDClassifier())
sgdc.train(data)
print("SGD Classifier: {:.2f}%".format(accuracy(sgdc, data) * 100))
save_model(sgdc, './models2.7/sgdc.pickle')

# Support Vector Classifier
svc = SklearnClassifier(SVC())
svc.train(data)
print('Support Vector Classifier trained.')
print("Support Vector Classifier: {:.2f}%".format(accuracy(svc, data) * 100))
save_model(svc, './models2.7/svc.pickle')

# Linear SV Classifier
lsvc = SklearnClassifier(LinearSVC())
lsvc.train(data)
print('Linear SV Classifier trained.')
print("Linear SV Classifier: {:.2f}%".format(accuracy(lsvc, data) * 100))
save_model(lsvc, './models2.7/lsvc.pickle')

# NuSV Classifier
nusvc = SklearnClassifier(NuSVC())
nusvc.train(data)
print('NuSV Classifier trained.')
print("NuSV Classifier: {:.2f}%".format(accuracy(nusvc, data) * 100))
save_model(nusvc, './models2.7/nusvc.pickle')
"""

# Decision Tree Classifier
dtc = SklearnClassifier(DecisionTreeClassifier())
dtc.train(data)
print('Decision Tree Classifier trained.')
print("Decision Tree Classifier: {:.2f}%".format(accuracy(dtc, data) * 100))
save_model(dtc, './models2.7/dtc.pickle')

# Random Forest Classifier
rfc = SklearnClassifier(RandomForestClassifier())
rfc.train(data)
print('Random Forest Classifier trained.')
print("Random Forest Classifier: {:.2f}%".format(accuracy(rfc, data) * 100))
save_model(rfc, './models2.7/rfc.pickle')

# End
end = timer()
print("Time taken:", end - start)
