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

def process_unseen_video(input_folder, output_folder, fps=1):
    unseen_video_folder = os.path.join(input_folder, 'unseen1')
    video_file = os.listdir(unseen_video_folder)[0]  # Assuming there's one video file in the 'unseen' folder
    video_path = os.path.join(unseen_video_folder, video_file)

    # Create a subfolder in output for this video
    video_name = os.path.splitext(video_file)[0]  # Get video name without extension
    video_output_folder = os.path.join(output_folder, 'unseen1', video_name)

    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    extract_frames(video_path, video_output_folder, fps=fps)

# Define the paths
input_folder = 'D:/DeepFakeDetection/Data'  # Folder containing your unseen video
output_folder = 'D:/DeepFakeDetection/DataFrames'  # Output folder for the extracted frames
fps = 10  # Extract 10 frames per second

# Process the unseen video
process_unseen_video(input_folder, output_folder, fps)
