import os
import numpy as np
import torch
from PoseRecognitionDataset import PoseRecognitionDataset
import argparse
import glob
import json
from scipy.interpolate import interp1d

def adjust_sequence_length(sequence_length, keypoints_seq):
    """Adjust sequence length using interpolation or sampling."""
    original_length = len(keypoints_seq)
    if original_length == 0:
        return np.zeros((sequence_length, 17, 2))  # For example, a sequence of zeros

    keypoints_seq = np.array(keypoints_seq)

    if original_length == sequence_length:
        return keypoints_seq
    elif original_length > sequence_length:
        indices = np.round(np.linspace(0, original_length - 1, sequence_length)).astype(int)
        return np.array([keypoints_seq[i] for i in indices])
    else:
        x = np.linspace(0, original_length - 1, num=original_length)
        x_new = np.linspace(0, original_length - 1, num=sequence_length)

        if keypoints_seq.ndim == 2:  # Case where the input array is 2D (frames, 34)
            interpolated_sequence = interp1d(x, keypoints_seq, axis=0, kind='linear')(x_new)
        elif keypoints_seq.ndim == 3:  # Normal case (frames, 17, 2)
            interpolated_sequence = np.array([interp1d(x, keypoints_seq[:, i, :], axis=0, kind='linear')(x_new) for i in range(17)])
            interpolated_sequence = interpolated_sequence.transpose(1, 0, 2)
        else:
            raise ValueError("Unexpected keypoints sequence shape: {}".format(keypoints_seq.shape))

        return interpolated_sequence

def create_embeddings(data_dir, model_path, sequence_length=30):
    # Load the entire model
    model = torch.load(model_path)
    model.eval()

    # Loop through each class in the data directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        with torch.no_grad():
            
            # Create an array to hold all embeddings for this class
            embeddings = []
            
            poses = glob.glob(f'{class_dir}/train/*.json')
            for p in poses:
                with open(p, 'r') as f:
                    pose = json.load(f)
                frame_dict = {}

                for person in pose:
                    frame_num = int(person['image_id'].split('.')[0])
                    box = person['box']
                    area = box[2] * box[3]  # Calculate area of the bounding box
                    keypoints = np.array(person['keypoints']).reshape(17, 3)[:, :2]  # Get x, y positions

                    # Normalize keypoints based on the bounding box
                    keypoints[:, 0] = (keypoints[:, 0] - box[0]) / box[2]  # Normalize x
                    keypoints[:, 1] = (keypoints[:, 1] - box[1]) / box[3]  # Normalize y

                    # Update the frame_dict to keep the keypoints of the person with the largest box
                    if frame_num not in frame_dict or area > frame_dict[frame_num]['area']:
                        frame_dict[frame_num] = {'keypoints': keypoints, 'area': area}

                # Collect the keypoints in order of frame number
                keypoints_seq = [frame_dict[frame]['keypoints'] for frame in sorted(frame_dict.keys())]
                keypoints_seq = torch.tensor(np.expand_dims(adjust_sequence_length(sequence_length, np.array(keypoints_seq)), 0), dtype=torch.float32)
                
                embedding = model(keypoints_seq).squeeze(0)  # Remove batch dimension
                print(f"embedding shape: {embedding.shape}")
                embeddings.append(embedding.numpy())

            # Convert list of embeddings to a numpy array and save it
            embeddings = np.array(embeddings)
            np.save(os.path.join(class_dir, 'embeddings.npy'), embeddings)
            print(f"Saved embeddings for class {class_name} in {os.path.join(class_dir, 'embeddings.npy')}")
                

        # # Load the dataset for this class
        # dataset = PoseRecognitionDataset(class_dir, sequence_length=sequence_length, split='train')
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # # Create an array to hold all embeddings for this class
        # embeddings = []

        # # Process each sample in the dataset
        # with torch.no_grad():
        #     for keypoints_seq, _ in dataloader:
        #         embedding = model(keypoints_seq).squeeze(0)  # Remove batch dimension
        #         print(f"embedding shape: {embedding.shape}")
        #         embeddings.append(embedding.numpy())

        # # Convert list of embeddings to a 2D numpy array and save it
        # embeddings = np.array(embeddings)  # This should be a 2D array: (num_samples, embedding_size)
        # np.save(os.path.join(class_dir, 'embeddings.npy'), embeddings)

        # # Convert list of embeddings to a numpy array and save it
        # embeddings = np.array(embeddings)
        # np.save(os.path.join(class_dir, 'embeddings.npy'), embeddings)
        # print(f"Saved embeddings for class {class_name} in {os.path.join(class_dir, 'embeddings.npy')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create embeddings for pose recognition.')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for pose data')

    args = parser.parse_args()

    create_embeddings(args.data_dir, args.model_path, args.sequence_length)

# python3 gait_recognition_create_embeddings.py --data_dir person_recognition_dataset --model_path runs_recognition/train2/weights/best.pt 