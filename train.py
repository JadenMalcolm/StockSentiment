import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = 100 * (np.array(all_labels) == np.array(all_predictions)).mean()
    print(f'Test Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down/Same (0)', 'Up (1)'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='red')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()
