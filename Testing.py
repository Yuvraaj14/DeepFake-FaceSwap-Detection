import torch
from CNN_LSTM import CNNLSTM  # Import the trained model class
from TestPreprocessing import UnseenVideoFrameDataset, transform  # Updated Dataset and preprocessing
from torch.utils.data import DataLoader

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the unseen dataset for testing
unseen_dataset = UnseenVideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames', sequence_length=10, transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)  # Batch size 1 for each video

# Evaluate unseen data
def evaluate_unseen_data(trained_model):
    trained_model.eval()  # Set model to evaluation mode
    results = []

    with torch.no_grad():  # No gradients needed for testing
        for idx, (images) in enumerate(unseen_loader):
            images = images.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted label (0 for Real, 1 for Fake)

            # Store the result
            results.append(predicted.item())
            predicted_label = "Real" if predicted.item() == 0 else "Fake"
            print(f'Video {idx+1}: Predicted - {predicted_label}')

    # Optional: If you need to calculate overall accuracy or other metrics
    # Calculate and display the overall results
    num_videos = len(results)
    print(f'Number of videos evaluated: {num_videos}')

if __name__ == "__main__":
    # Initialize the model (ensure the same architecture as training)
    model = CNNLSTM(input_channels=3, sequence_length=10, num_classes=2).to(device)

    # Load pre-trained weights if available (optional)
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))

    # Call the evaluation function
    evaluate_unseen_data(model)
