import re

import numpy as np
import spacy

nlp = spacy.load('en_core_web_lg')

custom_stop_words = ['ax', 'wa', 'ha', 'thi', 'hi']


def preprocess_text(text):

    text_clean = clean_text(text)

    text_nostop = remove_custom_stopwords(text_clean)

    return text_nostop


def clean_text(text):
    """
    Preprocess text for NLP applications: Lowercase, remove markups
    :param text: raw text
    :return: processed tokenized text
    """
    # Lowercase text
    text = text.lower()

    # Remove carriage returns and new lines
    text = re.sub('\r?\n|\r', ' ', text)

    # Remove Trademarks, Copyright, and Registered
    text = re.sub('(™|®|©|&trade;|&reg;|&copy;|&#8482;|&#174;|&#169;)', '', text)

    return text


def remove_custom_stopwords(text):
    """
    Removes custom stopwords from text
    :param text:
    :return:
    """

    tokens = spacy_tokenization(text)

    tokens_no_stop = [token for token in tokens if token not in custom_stop_words]

    return ' '.join(tokens_no_stop)


def spacy_tokenization(text):
    """
    Leverage SpaCy for tokenization
    :param text:
    :return:
    """

    doc = nlp(text)

    tokens = []

    for token in doc:
        if not token.is_punct:
            tokens.append(token.text)

    return tokens


def spacy_embedding(iterable_of_text, nlp=nlp):
    """
    Embed text with SpaCy
    :param iterable_of_text:
    :param nlp: SpaCy Model, default to en_core_web_lg
    :return: Mapped vectors from SpaCy simplex
    """

    counter = 0
    embeddings = []

    for iterable in iterable_of_text:
        embeddings.append(nlp(iterable).vector)

        counter += 1

        if counter % 100 == 0:
            print(counter)

    result = np.stack(embeddings, axis=0)

    return result
