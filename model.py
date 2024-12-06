import torch
import torch.nn as nn

class StockSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super(StockSentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim + 1, output_dim)  # Added 1 for sentiment
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, text, sentiment):
        # Ensure sentiment is on the same device as text
        sentiment = sentiment.to(text.device)

        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out_last = lstm_out[:, -1, :]
        lstm_out_last = self.layer_norm(lstm_out_last)
        lstm_out_last = self.dropout(lstm_out_last)
    
        # Combine LSTM output with sentiment
        combined_input = torch.cat((lstm_out_last, sentiment), dim=1)
    
        output = self.fc(combined_input)
        return output
