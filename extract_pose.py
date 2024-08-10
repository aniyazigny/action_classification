import sys
sys.path.append("/home/aniyazi/masters_degree/AlphaPose/")

import torch
from alphapose.models import builder
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.config import update_config
from alphapose.utils.vis import getTime
import cv2
import numpy as np

# Custom SimpleTransform that doesn't require a dataset with joint_pairs
class CustomSimpleTransform(SimpleTransform):
    def __init__(self, scale_factor, input_size, output_size, rot, sigma, train, gpu_device=None):
        super().__init__(None, scale_factor, False, input_size, output_size, rot, sigma, train, gpu_device)
        self._joint_pairs = []  # Provide an empty list or suitable default

def extract_poses_from_video(video_path, cfg, pose_model, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    input_size = cfg.DATA_PRESET.IMAGE_SIZE
    output_size = cfg.DATA_PRESET.HEATMAP_SIZE

    # Initialize the custom SimpleTransform with required parameters
    transform = CustomSimpleTransform(
        scale_factor=cfg.DATA_PRESET.get('SCALE_FACTOR', 0.25),  # Default scale factor
        input_size=input_size, 
        output_size=output_size,
        rot=cfg.DATA_PRESET.get('ROT_FACTOR', 30),  # Default rotation factor
        sigma=cfg.MODEL.get('SIGMA', 2),  # Default sigma value
        train=False,  # Set to False since we're doing inference
        gpu_device=device
    )

    pose_model.to(device)
    pose_model.eval()

    poses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        orig_img = frame
        img_k, img_k_for_net = transform(orig_img)

        with torch.no_grad():
            img_k_for_net = img_k_for_net.unsqueeze(0).to(device)
            output = pose_model(img_k_for_net)

            hm_to_coord = get_func_heatmap_to_coord(cfg)
            pred_coords, pred_scores = hm_to_coord(output[0], None)

            poses.append(pred_coords.cpu().numpy())

    cap.release()
    return poses

# Load the AlphaPose model
cfg = update_config('/home/aniyazi/masters_degree/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load('/home/aniyazi/masters_degree/AlphaPose/pretrained_models/fast_res50_256x192.pth', map_location='cuda'))

# Example usage
video_path = '/home/aniyazi/masters_degree/AlphaPose/runs/sample1/yatak_odasi.mp4'
poses = extract_poses_from_video(video_path, cfg, pose_model)
print()