from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

positive_file = open('./data/positive_reviews.txt', 'r')
negative_file = open('./data/negative_reviews.txt', 'r')

positive_reviews = [(review, 'pos') for review in positive_file.readlines()]
negative_reviews = [(review, 'neg') for review in negative_file.readlines()]
all_reviews = positive_reviews + negative_reviews
sia = SentimentIntensityAnalyzer()

right_preds = 0
for review in all_reviews:
    compound = sia.polarity_scores(review[0])['compound']
    if compound < 0 and review[1] == 'neg':
        right_preds += 1
    elif compound > 0 and review[1] == 'pos':
        right_preds += 1

print('Acc:', right_preds / len(all_reviews))