from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
toRemove = ['From:', 'To:', 'Cc:', 'Date:', 'Sent:', 'Subject:', 'Attachments:', 'Doc No', 'Case No', 'U.S.', 'UNCLASSIFIED', 'RELEASE', 'Department of State']
wnl = WordNetLemmatizer()

def nlpPipeLine(doc):
    # Tokenize
    tokens = tokenizer.tokenize(doc.lower())
    # Removes stop words
    stopped = [ word for word in tokens if not word in en_stop and len(word)>1 ]
    # Performs stemming
    lemmatized = [ wnl.lemmatize(word) for word in stopped ]
    return lemmatized

def rawEmailCleaner(doc):
    # Keeps only the lines that do not contain one of the words in the list toRemove
    rawText = ' '.join([line for line in doc.split('\n') if not sum([word in line for word in toRemove])])
    # Removes all uppercase words
    rawText = ' '.join([word.lower() for word in rawText.split(' ') if not word==word.upper()])
    return rawText

def processText(doc):
    return nlpPipeLine(rawEmailCleaner(doc))
