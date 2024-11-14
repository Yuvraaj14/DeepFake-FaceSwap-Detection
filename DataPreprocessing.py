import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for label in ['real', 'manipulated']:
            label_dir = os.path.join(self.root_dir, label)
            for video_folder in os.listdir(label_dir):
                video_dir = os.path.join(label_dir, video_folder)
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])  # Adjusted to .jpg
                data.append((video_dir, frames, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_dir, frames, label = self.data[idx]
        if len(frames) < self.sequence_length:
            frames = frames + [frames[-1]] * (self.sequence_length - len(frames))  # Padding if not enough frames
        
        frames = frames[:self.sequence_length]
        images = []
        
        for frame in frames:
            img_path = os.path.join(video_dir, frame)
            image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images)  # Stack the list of tensors into a single tensor
        label = 0 if label == 'real' else 1  # Assuming binary classification
        return images, label

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet standards
])

# Create dataset and dataloaders
train_dataset = VideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames/train', sequence_length=10, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = VideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames/test', sequence_length=10, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
