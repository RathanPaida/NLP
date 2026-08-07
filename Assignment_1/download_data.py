from datasets import load_dataset
import os
from huggingface_hub import hf_hub_download
import shutil

def download_indiccorp():
    print("Downloading IndicCorpV2 Telugu data...")
    try:
        # IndicCorpV2 is public, we can just download the text file directly
        file_path = hf_hub_download(repo_id="ai4bharat/IndicCorpV2", repo_type="dataset", filename="data/te.txt")
        shutil.copy(file_path, "te.txt")
        print("Downloaded te.txt successfully.")
    except Exception as e:
        print(f"Failed to download IndicCorpV2: {e}")

def download_oscar():
    print("\nDownloading OSCAR Telugu data...")
    print("Note: OSCAR is a gated dataset. Ensure you have accepted the terms and run 'huggingface-cli login' first.")
    try:
        # Load in streaming mode and write to file
        dataset = load_dataset("oscar-corpus/OSCAR-2301", "te", split="train", streaming=True)
        with open("oscar_te.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                if i >= 50000:  # Limit to 50k for assignment demo
                    break
                text = item.get('text', item.get('content', '')).strip()
                if text:
                    # Replace newlines with spaces so each paragraph/document is on one line
                    text = text.replace('\n', ' ')
                    f.write(text + "\n")
                if (i + 1) % 5000 == 0:
                    print(f"Downloaded {i+1} paragraphs from OSCAR...")
        print("Downloaded oscar_te.txt successfully.")
    except Exception as e:
        print(f"Failed to download OSCAR: {e}")

if __name__ == "__main__":
    download_indiccorp()
    download_oscar()
