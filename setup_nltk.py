# setup_nltk.py
import nltk
import os
import shutil
import sys

# --- Configuration ---
REQUIRED_RESOURCES = [
    {"name": "punkt", "path": "tokenizers/punkt"},
    {"name": "cmudict", "path": "corpora/cmudict"},
    {"name": "averaged_perceptron_tagger", "path": "taggers/averaged_perceptron_tagger"}
]

# This is the specific folder name the code is looking for.
EXPECTED_TAGGER_DIR_NAME = "averaged_perceptron_tagger_eng"
# This is the name NLTK actually downloads it as.
DOWNLOADED_TAGGER_DIR_NAME = "averaged_perceptron_tagger"


def download_nltk_data():
    """
    Downloads all necessary NLTK data resources and ensures they are correctly named.
    """
    print("--- Checking and Downloading NLTK Resources ---")

    for resource in REQUIRED_RESOURCES:
        try:
            nltk.data.find(resource['path'])
            print(f"'{resource['name']}' is already available.")
        except LookupError:
            print(f"Downloading '{resource['name']}'...")
            if not nltk.download(resource['name']):
                print(f"ERROR: Failed to download '{resource['name']}'. Please check your internet connection.", file=sys.stderr)
                sys.exit(1)
            print(f"'{resource['name']}' downloaded successfully.")

def apply_tagger_fix():
    """
    Finds where the perceptron tagger is located and creates the expected
    directory structure and files to prevent common LookupError and FileNotFoundError issues.
    """
    print("\n--- Applying NLTK path fix for 'averaged_perceptron_tagger_eng' ---")
    try:
        # Find the actual path to the downloaded resource's directory
        downloaded_dir_path = nltk.data.find(f"taggers/{DOWNLOADED_TAGGER_DIR_NAME}")
        
        # Define the path to the main data file within that directory
        source_pickle_path = os.path.join(downloaded_dir_path, f"{DOWNLOADED_TAGGER_DIR_NAME}.pickle")

        if not os.path.exists(source_pickle_path):
            print(f"ERROR: Source data file not found at '{source_pickle_path}'.", file=sys.stderr)
            return

        # Determine the parent 'taggers' directory
        taggers_dir = os.path.dirname(downloaded_dir_path)
        
        # Define the path where the application expects to find the tagger
        expected_dir_path = os.path.join(taggers_dir, EXPECTED_TAGGER_DIR_NAME)

        print(f"Source pickle file found at: {source_pickle_path}")
        print(f"Ensuring expected directory exists at: {expected_dir_path}")

        # Step 1: Create the expected directory if it doesn't exist
        os.makedirs(expected_dir_path, exist_ok=True)

        # Step 2: Define the exact file path from the error message
        expected_weights_json_path = os.path.join(expected_dir_path, f"{EXPECTED_TAGGER_DIR_NAME}.weights.json")

        print(f"Checking for expected file: {expected_weights_json_path}")

        # Step 3: Create the required file by copying the source pickle file.
        # This is a targeted fix for the FileNotFoundError, as some NLTK versions
        # incorrectly look for a .json file while the data is in the .pickle file.
        if not os.path.exists(expected_weights_json_path):
            print(f"Fix required. Copying source data to the expected path...")
            try:
                shutil.copyfile(source_pickle_path, expected_weights_json_path)
                print(f"Successfully created '{expected_weights_json_path}'")
                print("Path fix applied successfully.")
            except Exception as e:
                print(f"ERROR: Could not copy file to fix path. Error: {e}", file=sys.stderr)
        else:
            print("Required file already exists. No action needed.")

    except LookupError:
        print(f"ERROR: Could not find the '{DOWNLOADED_TAGGER_DIR_NAME}' resource.", file=sys.stderr)
        print("Please ensure the download step completed successfully.", file=sys.stderr)

if __name__ == "__main__":
    download_nltk_data()
    apply_tagger_fix()
    print("\n--- NLTK Setup Complete ---")
