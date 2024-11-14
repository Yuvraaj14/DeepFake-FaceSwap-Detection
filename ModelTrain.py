import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CNN_LSTM import CNNLSTM  # Ensure this import is correct based on your file structure
from DataPreprocessing import VideoFrameDataset, transform

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset and dataloaders
train_dataset = VideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames/train', sequence_length=10, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = VideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames/test', sequence_length=10, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model
model = CNNLSTM(input_channels=3, sequence_length=10, num_classes=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjust learning rate if needed

# Training function
def train_model(num_epochs=10, model_save_path='D:/DeepFakeDetection/Model_Files/CNN+LSTM_Model.pth'):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate model
        evaluate_model()

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == '__main__':
    # Set the path where the model will be saved
    model_save_path = 'D:/DeepFakeDetection/Model_Files/CNN+LSTM_Model.pth'
    train_model(num_epochs=10, model_save_path=model_save_path)  # Adjust number of epochs as needed
