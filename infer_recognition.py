import os
import json
import subprocess
import numpy as np
import torch
from scipy.spatial.distance import cdist
from PoseRecognitionDataset import PoseRecognitionDataset
from PoseRecognitionNetwork import PoseRecognitionNetwork
import argparse

def run_alphapose(video_path, alphapose_script, cfg_file, checkpoint_file, output_dir):
    # Run AlphaPose and save the results
    os.makedirs(output_dir, exist_ok=True)
    command = [
        'python', alphapose_script,
        '--cfg', cfg_file,
        '--checkpoint', checkpoint_file,
        '--video', video_path,
        '--outdir', output_dir
    ]
    print(f"Running AlphaPose with command: {command}")
    subprocess.run(command, check=True)

    # Find and return the JSON output from AlphaPose
    result_json = os.path.join(output_dir, 'alphapose-results.json')
    if not os.path.exists(result_json):
        raise FileNotFoundError(f"AlphaPose result JSON not found at {result_json}")
    
    return result_json

def load_embeddings(embeddings_dir):
    class_embeddings = {}
    for class_name in os.listdir(embeddings_dir):
        class_dir = os.path.join(embeddings_dir, class_name)
        embeddings_path = os.path.join(class_dir, 'embeddings.npy')
        if os.path.exists(embeddings_path):
            class_embeddings[class_name] = np.load(embeddings_path)
    return class_embeddings

def infer_pose(model, result_json, sequence_length=30):
    # Load JSON data (AlphaPose results)
    with open(result_json, 'r') as f:
        people_list = json.load(f)

    # Use PoseRecognitionDataset logic to process keypoints
    dataset = PoseRecognitionDataset(sequence_length=sequence_length)
    keypoints_seq = dataset._get_main_person_keypoints(people_list, result_json)
    keypoints_seq = dataset._adjust_sequence_length(np.array(keypoints_seq))

    # Convert to tensor and get the embedding
    inputs = torch.tensor(keypoints_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    embedding = model(inputs).squeeze(0).detach().numpy()  # Remove batch dimension and convert to numpy
    return embedding

def infer_recognition(input_path, model_path, embeddings_dir, alphapose_script, cfg_file, checkpoint_file, sequence_length=30, output_dir=None):
    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Load saved embeddings
    class_embeddings = load_embeddings(embeddings_dir)

    # Run AlphaPose if the input is a video, otherwise use the provided JSON
    if input_path.endswith('.json'):
        result_json = input_path
    else:
        result_json = run_alphapose(input_path, alphapose_script, cfg_file, checkpoint_file, output_dir)

    # Generate embeddings for the input pose
    input_embedding = infer_pose(model, result_json, sequence_length)

    # Ensure input_embedding is a 2D array (1, embedding_size) for comparison
    input_embedding = np.expand_dims(input_embedding, axis=0)

    # Find the closest match by comparing embeddings
    closest_class = None
    min_distance = float('inf')

    for class_name, embeddings in class_embeddings.items():
        # Ensure that the class embeddings are 2D
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        # Compute distances between input_embedding and class embeddings
        distances = cdist(input_embedding, embeddings, metric='euclidean')
        avg_distance = np.mean(distances)

        if avg_distance < min_distance:
            min_distance = avg_distance
            closest_class = class_name

    print(f"Input belongs to class: {closest_class} with distance: {min_distance}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose recognition based on saved embeddings using a trained model and AlphaPose.')

    parser.add_argument('--input_path', type=str, required=True, help='Path to the pose JSON or video file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='Directory containing class embeddings')
    parser.add_argument('--alphapose_script', type=str, default='/home/aniyazi/masters_degree/AlphaPose/scripts/demo_inference.py', help='Path to AlphaPose demo_inference.py script')
    parser.add_argument('--cfg_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/pretrained_models/fast_res50_256x192.pth', help='Path to the pretrained model')
    parser.add_argument('--output_dir', type=str, default='/home/aniyazi/masters_degree/action_classification/recognition_output', help='Directory to save AlphaPose outputs')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for pose data')

    args = parser.parse_args()

    infer_recognition(args.input_path, args.model_path, args.embeddings_dir, args.alphapose_script, args.cfg_file, args.checkpoint_file, args.sequence_length, args.output_dir)

# python3 /home/aniyazi/masters_degree/action_classification/infer_recognition.py --input_path /home/aniyazi/masters_degree/action_classification/person_recognition_dataset/babam/test/5.mp4 --model_path /home/aniyazi/masters_degree/action_classification/runs_recognition/train2/weights/best.pt --embeddings_dir /home/aniyazi/masters_degree/action_classification/person_recognition_dataset
