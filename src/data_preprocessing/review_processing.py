#!/usr/bin/env python

import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def get_wordnet_pos(treebank_tag):
    """Convert the part-of-speech naming scheme
       from the nltk default to WordNet lemmatizer 
       format"""

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_reviews(text_series, nltk_path=None):
    """
    Preprocess whiskey reviews:
        * removes alphanumerical words
        * normalizes strings
        * removes punctuation
        * removes stopwords
        * tokenization
        * lemmatization

    requires:
        text_series : Panda Series of Whisky Reviews
    returns:
        preprocessed series: a list of preprocessed reviews

    """

    # Strip spaces from top and end of string
    text_series = text_series.str.strip()

    # lowercase text
    text_series = text_series.str.lower()

    # remove punctuation
    text_series = text_series.str.replace('[^\w\s]', '')

    # replace all alphanumeric characters
    text_series = text_series.str.replace("\w*\d\w*", '')

    # tokenize words
    text_series = text_series.apply(nltk.word_tokenize)

    # remove stop words
    sw_filter = stopwords.words('english')
    text_series = text_series.apply(
        lambda x: [word for word in x if word not in sw_filter])

    # tag parts of speech
    processed_series = text_series.tolist()
    processed_series = nltk.pos_tag_sents(processed_series)
    processed_series = pd.Series(processed_series)
    processed_series = processed_series.apply(
        lambda x: [(y[0], get_wordnet_pos(y[1])) for y in x])

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    processed_series = processed_series.apply(
        lambda x: [lemmatizer.lemmatize(*y) for y in x])

    return processed_series.apply(lambda x: ' '.join(x))
