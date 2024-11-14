import cv2
import os

def extract_frames(video_path, output_folder, fps=1):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True

    while success:
        # Set the position of the video to the correct frame
        video.set(cv2.CAP_PROP_POS_MSEC, count * 1000 / fps)
        success, image = video.read()

        if success:
            # Save the frame as an image
            frame_filename = os.path.join(output_folder, f"frame{count:05d}.jpg")
            cv2.imwrite(frame_filename, image)
            count += 1

    video.release()

def process_videos(input_folder, output_folder, fps=1):
    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(input_folder, dataset_type)

        for category in ['real', 'manipulated']:
            category_path = os.path.join(dataset_path, category)
            output_category_folder = os.path.join(output_folder, dataset_type, category)

            if not os.path.exists(output_category_folder):
                os.makedirs(output_category_folder)

            for video_name in os.listdir(category_path):
                video_path = os.path.join(category_path, video_name)
                video_output_folder = os.path.join(output_category_folder, video_name.split('.')[0])

                if not os.path.exists(video_output_folder):
                    os.makedirs(video_output_folder)

                extract_frames(video_path, video_output_folder, fps=fps)

# Define the paths
input_folder = 'D:/DeepFakeDetection/Data'  # Folder containing your split videos
output_folder = 'D:/DeepFakeDetection/DataFrames'  # Output folder for the extracted frames
fps = 1  # Extract 1 frame per second

# Process the videos
process_videos(input_folder, output_folder, fps)
