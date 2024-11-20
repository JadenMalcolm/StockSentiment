import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 20
def full_corpus(file_path, pickle_path='full_corpus.pkl'):
    try:
        with open(pickle_path, 'rb') as file:
            corpus = pickle.load(file)
        print("Loaded full corpus from pickle.")
    except FileNotFoundError:
        print("Pickle file not found. Preprocessing corpus")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df = df.dropna()
        full_headlines = df.iloc[:, 3:28].astype(str).apply(' '.join, axis=1)
        corpus = preprocess_corpus(full_headlines)
        with open(pickle_path, 'wb') as file:
            pickle.dump(corpus, file)
        print(f"Full corpus pickled at {pickle_path}.")

    return corpus

def preprocess_corpus(corpus):
    if isinstance(corpus, str):
        corpus = [corpus]

    cleaned_corpus = []
    for headline in corpus:
        words = headline.split()
        words = [word for word in words if word not in stop_words]
        words = [ps.stem(word) for word in words]
        cleaned_corpus.append(' '.join(words))
    return cleaned_corpus

def prepare_data(train_corpus, test_corpus):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_corpus)

    X_train_seq = tokenizer.texts_to_sequences(train_corpus)
    X_test_seq = tokenizer.texts_to_sequences(test_corpus)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

    vocab_size = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1

    return X_train_pad, X_test_pad, vocab_size

def prepare_single_input(input_text, corpus):
    cleaned_text = preprocess_corpus(input_text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(corpus)
    input_seq = tokenizer.texts_to_sequences(cleaned_text)
    input_pad = pad_sequences(input_seq, maxlen=MAX_SEQUENCE_LENGTH)
    input_tensor = torch.tensor(input_pad, dtype=torch.long)

    return input_tensor

def load_and_preprocess_data(file_path, pickle_path='loader.pkl'):
    try:
        with open(pickle_path, 'rb') as file:
            X_train_pad, y_train, X_test_pad, y_test, vocab_size = pickle.load(file)
        print("Loaded preprocessed data from pickle.")
    
    except FileNotFoundError:
        print("Pickle file not found. Preprocessing corpus")

        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df = df.dropna()

        train = df[df['Date'] < '20150101']
        test = df[df['Date'] > '20141231']
        
        y_train = train['Label'].values
        y_test = test['Label'].values

        train_headlines = train.iloc[:, 3:28].astype(str).apply(' '.join, axis=1)
        test_headlines = test.iloc[:, 3:28].astype(str).apply(' '.join, axis=1)

        train_corpus = preprocess_corpus(train_headlines)
        test_corpus = preprocess_corpus(test_headlines)

        X_train_pad, X_test_pad, vocab_size = prepare_data(train_corpus, test_corpus)

        with open(pickle_path, 'wb') as file:
            pickle.dump((X_train_pad, y_train, X_test_pad, y_test, vocab_size), file)
        print(f"Preprocessed data pickled at {pickle_path}.")

    return torch.tensor(X_train_pad, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(X_test_pad, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32), vocab_size


def words_in_corpus(words_to_check, corpus):
    ps = PorterStemmer()

    stemmed_words = [ps.stem(word) for word in words_to_check]
    
    word_presence = {word: False for word in words_to_check}
    
    for doc in corpus:
        doc_words = doc.split() 
        for original_word, stemmed_word in zip(words_to_check, stemmed_words):
            if stemmed_word in doc_words:
                word_presence[original_word] = True 
    
    return word_presence
