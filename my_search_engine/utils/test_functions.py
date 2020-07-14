from unittest import TestCase
from my_search_engine.utils import functions


class Test(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_clean_text(self):
        self.assertEqual(functions.clean_text(text='hello world!'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello\nWorld!'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello\rWorld!'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!™'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!®'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!©'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!©®'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&trade;'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&reg;'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&copy;'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&#8482;'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&#174;'), 'hello world!')
        self.assertEqual(functions.clean_text(text='Hello World!&#169;'), 'hello world!')

    def test_remove_custom_stopwords(self):
        self.assertEqual(functions.remove_custom_stopwords(text='Hello World!'), 'Hello World')
        self.assertEqual(functions.remove_custom_stopwords(text='U.S.'), 'U.S.')
        self.assertEqual(functions.remove_custom_stopwords(text='DoD'), 'DoD')
        self.assertEqual(functions.remove_custom_stopwords(text='Hello'), 'Hello')
        self.assertEqual(functions.remove_custom_stopwords(text=' '.join(functions.custom_stop_words)), '')

    def test_spacy_tokenization(self):
        self.assertEqual(functions.spacy_tokenization(text='Hello World!'), ['Hello', 'World'])
        self.assertEqual(functions.spacy_tokenization(text='U.S.'), ['U.S.'])

    def test_preprocess_text(self):
        self.assertEqual(functions.preprocess_text(text='Hello World!'), 'hello world')
        self.assertEqual(functions.preprocess_text(text=' '.join(functions.custom_stop_words)), '')
