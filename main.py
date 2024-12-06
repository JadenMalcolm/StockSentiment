import os
import warnings
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from preprocess import load_and_preprocess_data, full_corpus, words_in_corpus, prepare_single_input
from model import StockSentimentLSTM
from train import train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # please shut up tensorflow
warnings.simplefilter('ignore', category=FutureWarning)  # I will not be setting weights to false

embedding_dim = 200
hidden_dim = 256
output_dim = 1
num_layers = 3
batch_size = 32
epochs = 40
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file = 'with_vader.csv'
X_train, y_train, sentiment_train, X_test, y_test, sentiment_test, vocab_size = load_and_preprocess_data(data_file)

# Ensure tensors are properly shaped and moved to the same device
sentiment_train = torch.tensor(sentiment_train, dtype=torch.float32).unsqueeze(1).to(device)
sentiment_test = torch.tensor(sentiment_test, dtype=torch.float32).unsqueeze(1).to(device)

# Combine X_train with sentiment_train and X_test with sentiment_test
train_data = TensorDataset(X_train, y_train, sentiment_train)
test_data = TensorDataset(X_test, y_test, sentiment_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
model = StockSentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model_path = 'stock_sentiment.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

train_model(model, train_loader, criterion, optimizer, device, epochs)

torch.save(model.state_dict(), 'stock_sentiment.pth')

evaluate_model(model, device, test_loader)

corpus = full_corpus(data_file)

words_to_check = ["stock", "really", "good", "news", "bananana", "ChatGPT", "LLM"]

result = words_in_corpus(words_to_check, corpus)

for word, is_present in result.items():
    if is_present:
        print(f"'{word}' is in the corpus.")
    else:
        print(f"'{word}' is not in the corpus.")

input_text = "Pilgrim knows how to progress"
sentiment_score = analyzer.polarity_scores(input_text)['compound']
sentiment_tensor = torch.tensor([[sentiment_score]], dtype=torch.float32).to(device)

input_tensor = prepare_single_input(input_text, corpus, device)
print(f"Input Tensor shape: {input_tensor.shape}")
print(f"Sentiment Tensor shape: {sentiment_tensor.shape}")
input_tensor = input_tensor.to(device)
print(input_tensor)
print(sentiment_tensor)
with torch.no_grad():
    output = model(input_tensor, sentiment_tensor)
print(output)

predicted_label = (output > 0).float()

print(f"Predicted label: {predicted_label.item()}")
