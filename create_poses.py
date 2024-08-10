import os
import shutil
import subprocess
import argparse

def create_custom_dataset(k400_train_dir, selected_classes, output_dir, alphapose_script, cfg_file, checkpoint_file):
    # Create the output directories if they don't exist
    clips_dir = os.path.join(output_dir, 'clips')
    poses_dir = os.path.join(output_dir, 'poses')
    
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    for class_name in selected_classes:
        class_videos_dir = os.path.join(k400_train_dir, class_name)
        if not os.path.exists(class_videos_dir):
            print(f"Class folder {class_videos_dir} does not exist, skipping...")
            continue

        # Create class subfolders in clips and poses directories
        class_clips_dir = os.path.join(clips_dir, class_name)
        class_poses_dir = os.path.join(poses_dir, class_name)
        
        os.makedirs(class_clips_dir, exist_ok=True)
        os.makedirs(class_poses_dir, exist_ok=True)

        # Copy videos and run AlphaPose inference
        for video_file in os.listdir(class_videos_dir):
            video_src_path = os.path.join(class_videos_dir, video_file)
            video_dst_path = os.path.join(class_clips_dir, video_file)

            # Copy the video to the clips directory
            shutil.copy(video_src_path, video_dst_path)
            print(f"Copied {video_file} to {class_clips_dir}")

            # Run AlphaPose inference and save the pose data
            outdir = class_poses_dir
            command = [
                'python', alphapose_script,
                '--cfg', cfg_file,
                '--checkpoint', checkpoint_file,
                '--video', video_src_path,
                '--outdir', outdir
            ]
            print(f"-------------\n\n---------COMMAND---------\n{command}\n\n-------------")

            print(f"Running AlphaPose on {video_file}...")
            subprocess.run(command)
            print(f"Finished processing {video_file}")

            # Rename the output JSON file to match the video file name
            pose_file_src = os.path.join(outdir, 'alphapose-results.json')
            pose_file_dst = os.path.join(outdir, f"{os.path.splitext(video_file)[0]}.json")
            if os.path.exists(pose_file_src):
                os.rename(pose_file_src, pose_file_dst)
                print(f"Renamed pose file to {pose_file_dst}")
            else:
                print(f"Error: Expected pose file {pose_file_src} not found!")

    print("Dataset creation and pose extraction completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a custom dataset and extract poses using AlphaPose.')

    # Define arguments with default values
    parser.add_argument('--input_dir', type=str, default='/home/aniyazi/masters_degree/kinetics-dataset/k400/videos/train', help='Path to Kinetics-400 train directory')
    parser.add_argument('--selected_classes', type=str, nargs='+', default=['deadlifting', 'exercising arm', 'front raises', 'jogging', 'lunge', 'punching bag', 'push up', 'situp'], help='List of class names to include')
    parser.add_argument('--output_dir', type=str, default='/home/aniyazi/masters_degree/action_recognition/data/train', help='Path to the output directory')
    parser.add_argument('--alphapose_script', type=str, default='/home/aniyazi/masters_degree/AlphaPose/scripts/demo_inference.py', help='Path to the demo_inference.py script')
    parser.add_argument('--cfg_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/pretrained_models/fast_res50_256x192.pth', help='Path to the pretrained model')

    # Parse arguments
    args = parser.parse_args()

    # Run the process
    create_custom_dataset(args.input_dir, args.selected_classes, args.output_dir, args.alphapose_script, args.cfg_file, args.checkpoint_file)


#sample run command

# python ~/masters_degree/action_recognition/create_poses.py --input_dir /home/aniyazi/masters_degree/kinetics-dataset/k400/videos/test --output_dir /home/aniyazi/masters_degree/action_recognition/data/test/
