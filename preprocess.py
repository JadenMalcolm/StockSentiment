import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

MAX_NB_WORDS = 10000 
MAX_SEQUENCE_LENGTH = 50  

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

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    df.dropna()
    # Used like this to implement the idea of a future to the model
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231']
    
    y_train = train['Label'].values
    y_test = test['Label'].values

    train_headlines = train.iloc[:, 3:28].astype(str).apply(' '.join, axis=1)
    test_headlines = test.iloc[:, 3:28].astype(str).apply(' '.join, axis=1)

    train_corpus = preprocess_corpus(train_headlines)
    test_corpus = preprocess_corpus(test_headlines)

    X_train_pad, X_test_pad, vocab_size = prepare_data(train_corpus, test_corpus)
    
    return torch.tensor(X_train_pad), torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(X_test_pad), torch.tensor(y_test, dtype=torch.float32), vocab_size
