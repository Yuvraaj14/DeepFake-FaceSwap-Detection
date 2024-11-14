import os
import shutil
from sklearn.model_selection import train_test_split

def split_videos(input_folder, output_folder, test_size=0.2):
    for category in ['real', 'manipulated']:
        category_path = os.path.join(input_folder, category)
        videos = os.listdir(category_path)

        # Split the videos into train and test sets
        train_videos, test_videos = train_test_split(videos, test_size=test_size, random_state=42)

        for video_set, folder_type in [(train_videos, 'train'), (test_videos, 'test')]:
            output_category_folder = os.path.join(output_folder, folder_type, category)
            if not os.path.exists(output_category_folder):
                os.makedirs(output_category_folder)

            for video_name in video_set:
                video_path = os.path.join(category_path, video_name)
                output_video_path = os.path.join(output_category_folder, video_name)
                shutil.copy(video_path, output_video_path)

# Define the paths
input_folder = 'D:/DeepFakeDetection/faceforensics++'  # Folder containing your original videos
output_folder = 'D:/DeepFakeDetection/Data'  # Output folder for the split data
test_size = 0.2  # 20% of the data for testing

# Split the videos
split_videos(input_folder, output_folder, test_size)
