import os
import subprocess
import argparse

def create_custom_dataset(input_dir, selected_classes, alphapose_script, cfg_file, checkpoint_file):
    for class_name in selected_classes:
        class_dir = os.path.join(input_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Class folder {class_dir} does not exist, skipping...")
            continue
        
        for split in ['train', 'validation', 'test']:
            split_dir = os.path.join(class_dir, split)
            if not os.path.exists(split_dir):
                print(f"Split folder {split_dir} does not exist, skipping...")
                continue

            for video_file in os.listdir(split_dir):
                video_src_path = os.path.join(split_dir, video_file)
                
                if not video_file.endswith(('.mp4', '.avi', '.mov')):  # Check for video formats
                    print(f"Skipping non-video file {video_file}")
                    continue

                # Run AlphaPose inference and save the pose data
                outdir = split_dir  # Output the poses in the same directory as the video
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

                # Rename the output JSON file to match the video file name but with .json extension
                pose_file_src = os.path.join(outdir, 'alphapose-results.json')
                pose_file_dst = os.path.join(outdir, f"{os.path.splitext(video_file)[0]}.json")
                if os.path.exists(pose_file_src):
                    os.rename(pose_file_src, pose_file_dst)
                    print(f"Renamed pose file to {pose_file_dst}")
                else:
                    print(f"Error: Expected pose file {pose_file_src} not found!")

    print("Pose extraction completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract poses using AlphaPose.')

    # Define arguments with default values
    parser.add_argument('--input_dir', type=str, default='/path/to/data', help='Path to the main data directory')
    parser.add_argument('--selected_classes', type=str, nargs='+', default=['class1', 'class2'], help='List of class names to include')
    parser.add_argument('--alphapose_script', type=str, default='/home/aniyazi/masters_degree/AlphaPose/scripts/demo_inference.py', help='Path to the demo_inference.py script')
    parser.add_argument('--cfg_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint_file', type=str, default='/home/aniyazi/masters_degree/AlphaPose/pretrained_models/fast_res50_256x192.pth', help='Path to the pretrained model')

    # Parse arguments
    args = parser.parse_args()

    # Run the process
    create_custom_dataset(args.input_dir, args.selected_classes, args.alphapose_script, args.cfg_file, args.checkpoint_file)
