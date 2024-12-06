import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for texts, labels, sentiments in train_loader:
            texts, labels, sentiments = texts.to(device), labels.to(device), sentiments.to(device)
            optimizer.zero_grad()
            outputs = model(texts, sentiments)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, device, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for texts, labels, sentiments in test_loader:
            texts, labels, sentiments = texts.to(device), labels.to(device), sentiments.to(device)
            outputs = model(texts, sentiments)
            predictions = torch.sigmoid(outputs).round()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.show()
