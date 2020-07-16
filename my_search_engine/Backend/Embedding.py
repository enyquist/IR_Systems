"""
Embedding preprocessed data
"""
import pickle

from my_search_engine.utils.functions import spacy_embedding

#######################################################################################################################
# Load data
#######################################################################################################################

with open('my_search_engine/Resources/newsgroups_train_preprocessed.pkl', 'rb') as file:
    newsgroups_train_preprocessed = pickle.load(file)

with open('my_search_engine/Resources/newsgroups_test_preprocessed.pkl', 'rb') as file:
    newsgroups_test_preprocessed = pickle.load(file)

#######################################################################################################################
# Embedding
#######################################################################################################################

newsgroups_train_embeddings = spacy_embedding(newsgroups_train_preprocessed)

newsgroups_test_embeddings = spacy_embedding(newsgroups_test_preprocessed)

#######################################################################################################################
# Save output
#######################################################################################################################

with open('my_search_engine/Resources/newsgroups_train_embeddings.pkl', 'wb') as file:
    pickle.dump(newsgroups_train_embeddings, file)

with open('my_search_engine/Resources/newsgroups_test_embeddings.pkl', 'wb') as file:
    pickle.dump(newsgroups_test_embeddings, file)
