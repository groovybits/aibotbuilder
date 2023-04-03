#!/usr/local/bin/python3
#
# Christi Kennedy April 1st 2023
#   store image fingerprints in a vector db using opencv, pinecone and perceptual hashes
#
# Insert an image into the Pinecone database
#   python video_analysis.py --insert --image path/to/image.jpg --metadata '{"filename": "image.jpg", "description": "A sample image"}'

# Process a video and compare it to a reference image
#   python video_analysis.py --process --image path/to/reference_image.jpg --video path/to/video.mp4 --output output --interval 30 --threshold 10

import argparse
import cv2
import json
import os
import pinecone
from PIL import Image
from imagehash import phash

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_namespace = "image_hashes"

def init_pinecone():
    pinecone.deinit()
    pinecone.init(api_key=pinecone_api_key)

def deinit_pinecone():
    pinecone.deinit()

def get_image_phash(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return phash(Image.fromarray(img_gray))

def phash_to_vector(phash_value):
    binary_string = format(phash_value.hash, "064b")
    return [float(bit) for bit in binary_string]

def insert_image(image_path, metadata):
    image_phash = get_image_phash(image_path)
    image_vector = phash_to_vector(image_phash)
    metadata_json = json.dumps(metadata)

    init_pinecone()
    namespaces = pinecone.list_namespaces()
    if pinecone_namespace not in namespaces:
        pinecone.create_namespace(pinecone_namespace)
    index = pinecone.Index(index_name=pinecone_namespace)
    index.upsert(items={str(image_phash): (image_vector, metadata_json)})
    deinit_pinecone()

    print(f"Image '{image_path}' inserted into the Pinecone database.")

def process_video(reference_image_path, video_path, output_path, frame_interval, hamming_threshold):
    reference_phash = get_image_phash(reference_image_path)
    similar_frames = []

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_number = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_phash = phash(Image.fromarray(img_gray))
            hamming_distance = reference_phash - frame_phash

            if hamming_distance <= hamming_threshold:
                time = frame_number // fps
                similar_frames.append((frame_number, time, hamming_distance))
                output_frame_path = os.path.join(output_path, f"frame_{frame_number}.jpg")
                cv2.imwrite(output_frame_path, frame)
                print(f"Found a similar frame ({frame_number}) at time {time} seconds with Hamming distance {hamming_distance}")

        frame_number += 1

    video.release()
    print(f"\nSimilar frames: {len(similar_frames)}")
    print("\nFrame Number, Time (s), Hamming Distance")
    for frame_info in similar_frames:
        print(f"{frame_info[0]}, {frame_info[1]}, {frame_info[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Analysis Tool")
    parser.add_argument("-i", "--insert", action="store_true", help="Insert image into the Pinecone database")
    parser.add_argument("-p", "--process", action="store_true", help="Process video and compare to reference image")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--metadata", type=str, help="Metadata for the image in JSON format")
    parser.add_argument("--video", type=str, help="Path to the video file")
    parser.add_argument("--output", type=str, default="output", help="Path to save similar frames")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval for processing")
    parser.add_argument("--threshold", type=int, default=10, help="Hamming distance threshold for similarity")

    args = parser.parse_args()

    if args.insert:
        if not args.image or not args.metadata:
            parser.error("The --insert option requires --image and --metadata arguments.")
        metadata = json.loads(args.metadata)
        insert_image(args.image, metadata)
    elif args.process:
        if not args.image or not args.video:
            parser.error("The --process option requires --image and --video arguments.")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        process_video(args.image, args.video, args.output, args.interval, args.threshold)
    else:
        parser.error("Please provide either --insert or --process option.")


