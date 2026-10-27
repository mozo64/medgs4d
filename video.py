
import cv2
import os
import argparse

def images_to_video(image_folder, output_video, fps=30):
    # Get list of images sorted by name
    images = [img for img in os.listdir(image_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
    images.sort()

    if not images:
        print("No images found in the folder.")
        return

    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # "XVID" for .avi
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠Skipping {img_path}, unable to read.")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in a folder to a video.")
    parser.add_argument("--input_folder", required=True, help="Folder containing input images")
    parser.add_argument("--output_folder", required=True, help="Folder to save the output video")
    parser.add_argument("--output_name", required=True, help="Name of the output video file (e.g., video.mp4)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(args.output_folder, args.output_name)
    import sys

    print("Using Python:", sys.executable)

    images_to_video(args.input_folder, output_path, fps=args.fps)



