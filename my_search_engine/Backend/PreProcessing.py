"""
For preprocessing data for later applications
"""

import pickle

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from wordcloud import WordCloud

from my_search_engine.utils.functions import preprocess_text

#######################################################################################################################
# Load data
#######################################################################################################################

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))

newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'))

#######################################################################################################################
# Preprocess Data
#######################################################################################################################

newsgroups_train_preprocessed = [preprocess_text(text) for text in newsgroups_train.data]

newsgroups_test_preprocessed = [preprocess_text(text) for text in newsgroups_test.data]

#######################################################################################################################
# Word Clouds - Checking for corpus specific stopwords
#######################################################################################################################

# # Merge all text into one big string
# all_text = ' '.join(newsgroups_train_preprocessed)
#
# cloud_accomplishments = WordCloud(background_color='white').generate(all_text)
#
# plt.imshow(cloud_accomplishments, interpolation='bilinear')
# plt.axis('off')
# plt.show()

#######################################################################################################################
# Save output
#######################################################################################################################

with open('my_search_engine/Resources/newsgroups_train_preprocessed.pkl', 'wb') as file:
    pickle.dump(newsgroups_train_preprocessed, file)

with open('my_search_engine/Resources/newsgroups_test_preprocessed.pkl', 'wb') as file:
    pickle.dump(newsgroups_test_preprocessed, file)
