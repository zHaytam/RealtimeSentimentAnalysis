from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtilsd
from vaderSentiment.vaderSentiment2 import SentimentIntensityAnalyzer, NEGATE
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from kafka import KafkaProducer


class SentimentAnalyzer:
    """
    Performs a sentiment analysis using the models loaded with pickle
    """

    __MODELS__ = ['bnbc', 'dtc', 'lgc', 'lsvc', 'mnbc', 'nbc', 'rfc', 'sgdc']

    def __init__(self):
        self.__load_models()
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        self.__load_most_used_words()

    def __load_most_used_words(self):
        positive_file = open('./data/positive_reviews.txt', 'r')
        negative_file = open('./data/negative_reviews.txt', 'r')

        positive_reviews = [(review, 'pos') for review in positive_file.readlines()]
        negative_reviews = [(review, 'neg') for review in negative_file.readlines()]
        all_reviews = positive_reviews + negative_reviews

        all_words = []
        for review in all_reviews:
            all_words.extend(self.extract_words(review[0]))

        all_words = nltk.FreqDist(all_words)
        self.most_used_words = [w[0] for w in sorted(all_words.items(), key=lambda x: x[1], reverse=True)][:10000]

    def __load_models(self):
        self.models = []
        for model in self.__MODELS__:
            file = open('./models2.7/{}.pickle'.format(model), 'rb')
            self.models.append(pickle.load(file))
            file.close()

        print('{} models loaded.'.format(len(self.models)))

    @staticmethod
    def __lowercase_negation(text):
        for neg in NEGATE:
            text = re.sub('(?i)' + re.escape(neg), neg, text)

        return text

    def remove_stopwords(self, words):
        filtered = []

        for word in words:
            if word not in self.stopwords:
                filtered.append(word)

        return filtered

    def extract_words(self, text):
        words = [word for word in word_tokenize(text)]
        words = self.remove_stopwords(words)
        # words = [stemmer.stem(word) for word in words]
        words = [self.stemmer.stem(word.lower()) for word in words]
        return words

    def create_features(self, text):
        words = self.extract_words(text)
        features = dict([(muw, True if muw in words else False) for muw in self.most_used_words])
        return features

    def classify(self, text):
        """
        Creates the set of features for the text and performs the sentiment analysis with a vote
        Returns the classification and the confidence
        """
        features = self.create_features(text)
        votes = [model.classify(features) for model in self.models]
        choice = max(votes, key=votes.count)
        conf = float(votes.count(choice)) / len(votes)

        return [choice, conf]

    def polarity_scores(self, text):
        return self.sia.polarity_scores(SentimentAnalyzer.__lowercase_negation(text))

    def perform_sentiment_analysis(self, record):
        print(record)
        compound = self.polarity_scores(record[1])['compound']
        classification = self.classify(record[1])
        return record + [compound] + classification


sentiment_analyzer = SentimentAnalyzer()
sa_producer = KafkaProducer(bootstrap_servers='localhost:9092')


def send_rdd(rdd):
    records = rdd.collect()
    for record in records:
        record[1] = record[1].encode('utf-8')
        sa_producer.send('sa', '\t'.join([str(e) for e in record]))


sc = SparkContext(appName='SentimentAnalysis')
sc.setLogLevel('WARN')
ssc = StreamingContext(sc, 5)

kvs = KafkaUtils.createDirectStream(ssc, ['comments'], {
    'bootstrap.servers': 'localhost:9092',
    'auto.offset.reset': 'smallest'
})


# Parse comments and perform sentiment analysis
comments = kvs.map(lambda text: text[1].split('\t'))
sentiments = comments.map(lambda record: sentiment_analyzer.perform_sentiment_analysis(record))
sentiments.pprint()

# Send results through the producer
sentiments.foreachRDD(send_rdd)

ssc.start()
ssc.awaitTermination()
