import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnseenVideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        unseen_folder = os.path.join(self.root_dir, 'unseen1')  # Points to DataFrames/unseen
        # Assume there's only one video folder in unseen
        for video_folder in os.listdir(unseen_folder):
            video_folder_path = os.path.join(unseen_folder, video_folder)
            if os.path.isdir(video_folder_path):
                frames = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.jpg')])
                data.append((video_folder_path, frames))  # Add video folder path and its frames
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_dir, frames = self.data[idx]
        
        if len(frames) < self.sequence_length:
            frames = frames + [frames[-1]] * (self.sequence_length - len(frames))  # Pad if fewer frames
        
        frames = frames[:self.sequence_length]  # Ensure sequence length
        images = []
        
        for frame in frames:
            img_path = os.path.join(video_dir, frame)
            image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images)  # Stack the images as tensors
        return images

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet standards
])

# Create the dataset and dataloader for unseen data
unseen_dataset = UnseenVideoFrameDataset(root_dir='D:/DeepFakeDetection/DataFrames', sequence_length=10, transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# Now, you can use this unseen_loader to evaluate the model on your test video data.
