"""
Vectorizing preprocessed data
"""
import pickle

import numpy as np
import spacy

nlp = spacy.load('en_core_web_lg')

#######################################################################################################################
# Load data
#######################################################################################################################

with open('my_search_engine/Resources/newsgroups_train_preprocessed.pkl', 'rb') as file:
    newsgroups_train_preprocessed = pickle.load(file)

#######################################################################################################################
# Embedding
#######################################################################################################################

counter = 0

embeddings = []

for news in newsgroups_train_preprocessed:
    embeddings.append(nlp(news).vector)

    counter += 1

    if counter % 100 == 0:
        print(counter)

newsgroups_train_embeddings = np.stack(embeddings, axis=0)

#######################################################################################################################
# Save output
#######################################################################################################################

with open('my_search_engine/Resources/newsgroups_train_embeddings.pkl', 'wb') as file:
    pickle.dump(newsgroups_train_embeddings, file)
