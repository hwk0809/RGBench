from rgbench.visualization import visualize_pcds_in_directory
import os

if __name__ == "__main__":
    # --- User Configuration ---
    # For convenience, you can fill in the path to your data folder directly here.
    # For example: DEFAULT_PCD_DIRECTORY = "/path/to/your/project/segmeqqnt_pcds"
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DEFAULT_PCD_DIRECTORY = os.environ.get(
        "RGBENCH_PCD_DIRECTORY",
        os.path.join(REPO_ROOT, "data", "sample", "pcd"),
    )
    # --------------------------
    print("--- Point Cloud Sequence Visualizer ---")
    # If the default path is valid, use it directly; otherwise, prompt the user for input.
    if DEFAULT_PCD_DIRECTORY and os.path.isdir(DEFAULT_PCD_DIRECTORY):
        directory_to_check = DEFAULT_PCD_DIRECTORY
        print(f"Using preset directory: {directory_to_check}")
    else:
        directory_to_check = input("Please enter the path to the directory containing your '.pcd' files: ")

    # Strip any surrounding quotes from the path (useful when dragging and dropping a folder).
    directory_to_check = directory_to_check.strip().strip("'\"")

    # Call the main function
    visualize_pcds_in_directory(directory_to_check,autoplay=False,reverse=False,loop=False)