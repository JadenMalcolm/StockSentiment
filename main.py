import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # please shut up tensorflow
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from preprocess import load_and_preprocess_data
from model import StockSentimentLSTM
from train import train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences

embedding_dim = 50
hidden_dim = 64
output_dim = 1
num_layers = 2
batch_size = 32
epochs = 10
learning_rate = 0.001


data_file = 'stock_senti_analysis.csv'
X_train, y_train, X_test, y_test, vocab_size = load_and_preprocess_data(data_file)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
model = StockSentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model_path = 'stock_sentiment.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

train_model(model, train_loader, criterion, optimizer, epochs)

torch.save(model.state_dict(), 'stock_sentiment.pth')

evaluate_model(model, test_loader)
