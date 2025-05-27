import os
import shutil
from math import ceil

def split_and_merge_folders(base_folders, output_base_folder, num_parts=10):
    """
    Split data from each folder into `num_parts` parts and merge corresponding parts into new folders.

    Args:
        base_folders (list): List of input folder paths.
        output_base_folder (str): Base folder to store merged outputs.
        num_parts (int): Number of parts to split each folder into.
    """
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # Process each folder
    for folder in base_folders:
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue

        # Get all files in the folder
        all_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        total_files = len(all_files)
        
        if total_files == 0:
            print(f"Folder {folder} is empty.")
            continue

        # Calculate number of files per part
        files_per_part = ceil(total_files / num_parts)

        # Create subfolders and distribute files
        folder_name = os.path.basename(folder)
        for part in range(num_parts):
            part_folder = f"{folder_name}_{part}"
            part_folder_path = os.path.join(output_base_folder, part_folder)
            os.makedirs(part_folder_path, exist_ok=True)

            # Get files for this part
            start_idx = part * files_per_part
            end_idx = min(start_idx + files_per_part, total_files)
            for file_path in all_files[start_idx:end_idx]:
                shutil.copy(file_path, part_folder_path)

    # Merge corresponding parts into new folders
    for part in range(num_parts):
        new_folder = os.path.join(output_base_folder, f"new_input_{part}")
        os.makedirs(new_folder, exist_ok=True)

        for folder in base_folders:
            folder_name = os.path.basename(folder)
            part_folder = os.path.join(output_base_folder, f"{folder_name}_{part}")

            if os.path.exists(part_folder):
                for file_name in os.listdir(part_folder):
                    src_file = os.path.join(part_folder, file_name)
                    dst_file = os.path.join(new_folder, file_name)
                    shutil.copy(src_file, dst_file)

if __name__ == "__main__":
    # Define input folders
    base_folders = [
        "/data3/liupei/NDSS2026/TrainData/2CorrectedDataset/train_real_compilable",
        "/data3/liupei/NDSS2026/TrainData/2CorrectedDataset/train_real_simple_io",
        "/data3/liupei/NDSS2026/TrainData/2CorrectedDataset/train_synth_rich_io",
        "/data3/liupei/NDSS2026/TrainData/2CorrectedDataset/train_synth_simple_io",
    ]

    # Define output folder
    output_base_folder = "/data3/liupei/NDSS2026/TrainData/3SplitedDataset0"

    # Split and merge folders
    split_and_merge_folders(base_folders, output_base_folder, num_parts=80)
