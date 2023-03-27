#!/usr/local/bin/python3

## https://pjreddie.com/darknet/yolo/

## Christi Kennedy 2023

## Take a source and a derivative video of it,
## get the VMAF, the Perceptual Hashes, the hamming distances
## and information of each frames video stats and the
## opencv analysis of images in the frames. build a DB
## of the video with all of these stats to use as a training
## for adding into AI and analyzing videos for various characteristics
## contents, what objects are in them, quality values, correllations
## of perceptual hash fingerprints and hamming distance between quality
## and also objects behaviors. find scene changes and understand
## the video in many more dimensions

## WIP This is an experiment / research into video identification

import concurrent.futures
import os
import json
import argparse
import cv2
import imagehash
import glob
import ffmpeg
import np
from PIL import Image
import pytesseract
from imagehash import phash
import subprocess
from scipy.spatial.distance import hamming
import sys

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps
    cap.release()
    return duration

def process_frame(i, source_frame_path, derivative_frame_path, vmaf_score, yolov3_weights, yolov3_cfg, coco_names):
    source_frame = cv2.imread(source_frame_path)
    derivative_frame = cv2.imread(derivative_frame_path)
    source_frame_phash = compute_image_phash(source_frame_path)
    derivative_frame_phash = compute_image_phash(derivative_frame_path)
    source_phash_bin = bin(int(source_frame_phash, 16))[2:].zfill(64)
    derivative_phash_bin = bin(int(derivative_frame_phash, 16))[2:].zfill(64)
    hamming_distance = sum([1 for x, y in zip(source_phash_bin, derivative_phash_bin) if x != y])

    source_frame_description = recognize_objects(
                                source_frame,
                                yolov3_weights,
                                yolov3_cfg,
                                coco_names,
                            )

    source_frame_text = extract_text(source_frame)

    derivative_frame_description = recognize_objects(
                                derivative_frame,
                                yolov3_weights,
                                yolov3_cfg,
                                coco_names,
                            )

    derivative_frame_text = extract_text(derivative_frame)

    return {
        'frame': i,
        'source_frame_phash': source_frame_phash,
        'derivative_frame_phash': derivative_frame_phash,
        'hamming_distance': hamming_distance,
        'source_frame_description': source_frame_description,
        'source_frame_text': source_frame_text.strip(),
        'derivative_frame_description': derivative_frame_description,
        'derivative_frame_text': derivative_frame_text.strip(),
        'vmaf_score': vmaf_score,
    }

def extract_text(image):
    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract text from the image using pytesseract
    text = pytesseract.image_to_string(pil_image)

    return text

def compute_image_phash(image_path):
    image = Image.open(image_path)
    image_phash = phash(image)
    return str(image_phash)

def recognize_objects(image, model_path, config_path, class_names_path):
    # Read class names from file
    with open(class_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the YOLO model
    net = cv2.dnn.readNet(model_path, config_path)

    # Get layer names
    layer_names = net.getLayerNames()
    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #output_layers = [layer_names[i[0][0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Prepare input for the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run object detection
    outputs = net.forward(output_layers)

    # Initialize lists to store detection results
    class_ids = []
    confidences = []
    boxes = []

    # Process the outputs and filter detected objects
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, width, height = (
                    int(detection[0] * image.shape[1]),
                    int(detection[1] * image.shape[0]),
                    int(detection[2] * image.shape[1]),
                    int(detection[3] * image.shape[0]),
                )

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Generate a sentence describing the detected objects
    objects_detected = [classes[class_ids[i]] for i in np.asarray(indices).flatten()]
    objects_sentence = ", ".join(objects_detected) if objects_detected else "no objects"
    description = f"This frame contains {objects_sentence}."

    return description

def extract_frames(video_path, output_dir, start_seconds, end_seconds, suffix=''):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, f'{video_name}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Calculate the duration of the video segment to analyze
    duration = end_seconds - start_seconds

    cmd = (
        f'ffmpeg -i {video_path} -ss {start_seconds} -t {duration} -vf "fps=1" '
        f'{os.path.join(frames_dir, f"{video_name}{suffix}_%04d.png")}'
    )
    subprocess.run(cmd, shell=True, check=True, text=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, f'{video_name}{suffix}_*.png')))

    return frame_paths

def compute_perceptual_hashes(frames):
    phashes = []

    for frame in frames:
        pil_image = Image.fromarray(frame)
        phash = imagehash.phash(pil_image)
        phashes.append(phash)

    return phashes

def compute_vmaf(ref_video_path, video_path, start_seconds, end_seconds):
    # Get video dimensions
    width, height = get_video_dimensions(ref_video_path)

    # Calculate the duration of the video segment to analyze
    duration = end_seconds - start_seconds

    log_path = "vmaf_log.json"
    cmd = (
        f"ffmpeg -ss {start_seconds} -t {duration} -i {video_path} -ss {start_seconds} -t {duration} -i {ref_video_path} "
        f'-lavfi libvmaf="log_path={log_path}:log_fmt=json:psnr=1:ssim=1:ms_ssim=1" '
        f"-f null -"
    )
    subprocess.run(cmd, shell=True, check=True, text=True)

    with open(log_path) as f:
        vmaf_log_data = json.load(f)

    vmaf_scores = [frame["metrics"] for frame in vmaf_log_data["frames"]]
    return vmaf_scores

def get_video_dimensions(video_path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json {video_path}'
    output = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    output_json = json.loads(output.stdout)
    width = output_json['streams'][0]['width']
    height = output_json['streams'][0]['height']
    return width, height

def process_videos(source_video_path, derivative_video_path, output_dir, start_seconds, end_seconds):
    # Extract frames from the source and derivative videos
    print("Extracting frames from source video...")
    source_frame_paths = extract_frames(source_video_path, output_dir, start_seconds, end_seconds, suffix='_source')
    print("Extracting frames from derivative video...")
    derivative_frame_paths = extract_frames(derivative_video_path, output_dir, start_seconds, end_seconds, suffix='_derivative')

    # Compute VMAF scores
    print("Computing VMAF scores...")
    vmaf_scores = compute_vmaf(source_video_path, derivative_video_path, start_seconds, end_seconds)

    # Analyze frames
    results = []
    total_frames = len(source_frame_paths)
    # Analyzing frames
    print("Analyzing frames...")

    # In the process_videos function, replace the for loop with the following code:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_frame,
                i,
                source_frame_path,
                derivative_frame_path,
                vmaf_scores[i],
                '/Users/christi/src/aibotbuilder/yolov3.weights',
                '/Users/christi/src/aibotbuilder/yolov3.cfg',
                '/Users/christi/src/aibotbuilder/coco.names',
            )
            for i, (source_frame_path, derivative_frame_path) in enumerate(zip(source_frame_paths, derivative_frame_paths))
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Processed frame {result['frame']}")

    results.sort(key=lambda x: x['frame'])  # Sort the results by frame number

    print("\nSaving results to JSON file...")
    # Save results to a JSON file
    output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(derivative_video_path))[0]}_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames and extract data for AI training.')
    parser.add_argument('source_video_path', type=str, help='Path to the source video')
    parser.add_argument('derivative_video_path', type=str, help='Path to the derivative video')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--start', type=int, default=0, help='Start time in seconds')
    parser.add_argument('--end', type=int, default=None, help='End time in seconds')

    args = parser.parse_args()

    end = args.end
    if end is None:
        end = get_video_duration(args.source_video_path)

    process_videos(args.source_video_path, args.derivative_video_path, args.output_dir, args.start, end)
