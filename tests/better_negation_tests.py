from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, NEGATE
import re


def lowercase_negation(text):
    for neg in NEGATE:
        text = re.sub('(?i)' + re.escape(neg), neg, text)

    return text


example_sentences = ['Not bad at all',
                     "Isn't it bad to do this?",
                     'I do NOT like this..',
                     "THIS ISN'T EVEN LEGIT",
                     "Wouldn't it be better to just avoid all of this?",
                     'SUB TO ME YOU WILL NOT REGRET IT']

sia = SentimentIntensityAnalyzer()

for sentence in example_sentences:
    print(sia.polarity_scores(sentence)['compound'], 'vs',
          sia.polarity_scores(lowercase_negation(sentence))['compound'])

print('-' * 20)

