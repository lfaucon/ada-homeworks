
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pycountry
import nltk.sentiment.vader
from wordcloud import WordCloud

nb = 0

def cleanData(text):
    # tokenization, stemming + remove punctuation and stop words
    text = text.lower()

    # remove punctuation and stop words
    tknzr = nltk.RegexpTokenizer(r'\w+')
    listWords = tknzr.tokenize(text)
    
    # stemming
    stemmer = nltk.SnowballStemmer("english")

    listWordsClean = [stemmer.stem(word) for word in listWords]    
    return listWordsClean

def getSentimentsCountries(x):
    global nb
    # tokenization, stemming + remove punctuation and stop words
    text = x.RawText.lower()

    # remove punctuation and stop words
    tknzr = nltk.RegexpTokenizer(r'\w+')
    listWords = tknzr.tokenize(text)

    countries = list()
    for country in list(pycountry.countries):
        if country.name.lower() in listWords:
            countries.append(country.name)

    x['countries'] = countries
    score = analyzer.polarity_scores(' '.join(listWords))['compound']
    x['score'] = score
    print(nb*100/float(len(df_emails)))

    nb += 1

    return x


### read email database
filename = './hillary-clinton-emails/Emails.csv'
df_emails = pd.read_csv(filename)
df_emails.columns

text = ''

for index, row in df_emails.iterrows():
    text += str(row.ExtractedBodyText) + ' '

textClean = cleanData(text)


analyzer = nltk.sentiment.vader.SentimentIntensityAnalyzer('vader_lexicon.txt')
df_emails = df_emails.apply(getSentimentsCountries, axis=1)

