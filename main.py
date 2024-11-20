import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # please shut up tensorflow
warnings.simplefilter('ignore', category=FutureWarning) # I will not be setting weights to false
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from preprocess import load_and_preprocess_data
from model import StockSentimentLSTM
from train import train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocess import prepare_single_input, full_corpus, words_in_corpus


embedding_dim = 200
hidden_dim = 128
output_dim = 1
num_layers = 3
batch_size = 32
epochs = 10
learning_rate = 0.0005


data_file = 'stock_senti_analysis.csv'
X_train, y_train, X_test, y_test, vocab_size = load_and_preprocess_data(data_file)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
model = StockSentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model_path = 'stock_sentiment.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

# train_model(model, train_loader, criterion, optimizer, epochs)

torch.save(model.state_dict(), 'stock_sentiment.pth')

evaluate_model(model, test_loader)


corpus = full_corpus(data_file)

words_to_check = ["stock", "really", "good", "news", "bananana", "ChatGPT", "LLM"]

result = words_in_corpus(words_to_check, corpus)

for word, is_present in result.items():
    if is_present:
        print(f"'{word}' is in the corpus.")
    else:
        print(f"'{word}' is not in the corpus.")

input_text = "Deadline-day deals,The minds are still willing but what about the flesh?"

input_tensor = prepare_single_input(input_text, corpus)
print(input_tensor)


with torch.no_grad():
    output = model(input_tensor)
    print(output)


predicted_label = (output > 0).float()

print(f"Predicted label: {predicted_label.item()}")

