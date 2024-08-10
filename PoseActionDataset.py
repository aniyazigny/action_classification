import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.interpolate import interp1d

class PoseActionDataset(Dataset):
    def __init__(self, poses_dir=None, sequence_length=30, keypoints_seq=None):
        self.poses_dir = poses_dir
        self.sequence_length = sequence_length
        self.data = []
        self.class_labels = []

        # If keypoints_seq is provided, use it directly
        if keypoints_seq is not None:
            self.keypoints_seq = keypoints_seq
        elif self.poses_dir:
            # Automatically detect class names from folder names in poses_dir
            self._prepare_file_list()

    def _prepare_file_list(self):
        """Prepare list of JSON file paths and their associated class labels."""
        for class_name in os.listdir(self.poses_dir):
            class_dir = os.path.join(self.poses_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_labels.append(class_name)
                label = self.class_labels.index(class_name)
                for json_file in os.listdir(class_dir):
                    json_path = os.path.join(class_dir, json_file)
                    self.data.append((json_path, label))

    def _get_main_person_keypoints(self, people_list, json_path):
        """Extract and normalize keypoints for the person with the largest bounding box in each frame."""
        frame_dict = {}

        for person in people_list:
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
        keypoints_sequence = [frame_dict[frame]['keypoints'] for frame in sorted(frame_dict.keys())]

        if len(keypoints_sequence) == 0:
            print(f"Warning: Empty keypoints sequence found in file {json_path}")

        return keypoints_sequence

    def _adjust_sequence_length(self, keypoints_seq):
        """Adjust sequence length using interpolation or sampling."""
        original_length = len(keypoints_seq)
        if original_length == 0:
            # Handle empty sequence case, you could return a sequence of zeros or skip this example
            return np.zeros((self.sequence_length, 17, 2))  # For example, a sequence of zeros

        keypoints_seq = np.array(keypoints_seq)  # Ensure keypoints_seq is an array

        if original_length == self.sequence_length:
            return keypoints_seq
        elif original_length > self.sequence_length:
            indices = np.round(np.linspace(0, original_length - 1, self.sequence_length)).astype(int)
            return np.array([keypoints_seq[i] for i in indices])
        else:
            x = np.linspace(0, original_length - 1, num=original_length)
            x_new = np.linspace(0, original_length - 1, num=self.sequence_length)

            # Ensure the keypoints_seq is properly shaped before interpolation
            keypoints_seq = np.array(keypoints_seq)
            if keypoints_seq.ndim == 2:  # Case where the input array is 2D (frames, 34)
                interpolated_sequence = interp1d(x, keypoints_seq, axis=0, kind='linear')(x_new)
            elif keypoints_seq.ndim == 3:  # Normal case (frames, 17, 2)
                interpolated_sequence = np.array([interp1d(x, keypoints_seq[:, i, :], axis=0, kind='linear')(x_new) for i in range(17)])
                interpolated_sequence = interpolated_sequence.transpose(1, 0, 2)
            else:
                raise ValueError("Unexpected keypoints sequence shape: {}".format(keypoints_seq.shape))

            return interpolated_sequence

    def __getitem__(self, idx):
        json_path, label = self.data[idx]

        # Load JSON file
        with open(json_path, 'r') as f:
            people_list = json.load(f)

        # Get keypoints for the main person across all frames
        keypoints_seq = self._get_main_person_keypoints(people_list, json_path)

        # Adjust sequence length
        keypoints_seq = self._adjust_sequence_length(np.array(keypoints_seq))

        return torch.tensor(keypoints_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     json_path, label = self.data[idx]

    #     # Load JSON file
    #     with open(json_path, 'r') as f:
    #         people_list = json.load(f)

    #     # Get keypoints for the main person across all frames
    #     keypoints_seq = self._get_main_person_keypoints(people_list)

    #     # Adjust sequence length
    #     keypoints_seq = self._adjust_sequence_length(np.array(keypoints_seq))

    #     return torch.tensor(keypoints_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    poses_dir = '/home/aniyazi/masters_degree/action_recognition/data/train/poses/'
    dataset = PoseActionDataset(poses_dir, sequence_length=30)

    # Print the automatically detected class labels
    print("Detected classes:", dataset.class_labels)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print()
