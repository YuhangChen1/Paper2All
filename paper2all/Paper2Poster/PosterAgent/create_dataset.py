from datasets import load_dataset
import os
import subprocess

from PIL import Image
import json

def generate_meta_json(base_dir='Paper2Poster-data'):
    # Loop over each item in the specified base directory
    for folder_name in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, folder_name)
        
        # Ensure the item is a directory
        if os.path.isdir(subfolder_path):
            poster_path = os.path.join(subfolder_path, 'poster.png')
            
            # Check if the poster.png exists in the subfolder
            if os.path.exists(poster_path):
                try:
                    # Open the image and get size (width, height)
                    with Image.open(poster_path) as img:
                        width, height = img.size
                    
                    # Prepare metadata dictionary
                    metadata = {
                        'width': width,
                        'height': height
                    }
                    
                    # Write metadata to meta.json in the same subfolder
                    meta_json_path = os.path.join(subfolder_path, 'meta.json')
                    with open(meta_json_path, 'w') as json_file:
                        json.dump(metadata, json_file)
                    
                    print(f"Metadata for '{folder_name}' saved successfully.")
                except Exception as e:
                    print(f"Error processing image in folder '{folder_name}': {e}")
            else:
                print(f"No poster.png found in folder '{folder_name}'.")

if __name__ == "__main__":
    dataset = load_dataset("Paper2Poster/Paper2Poster", split="train")
    os.makedirs('Paper2Poster-data', exist_ok=True)
    for data in dataset:
        paper_title = data['title']
        paper_url = data['paper_url']
        poster_url = data['image_url']
        qa = data['qa']

        os.makedirs(f'Paper2Poster-data/{paper_title}', exist_ok=True)

        paper_output_path = os.path.join('Paper2Poster-data', paper_title, 'paper.pdf')
        poster_output_path = os.path.join('Paper2Poster-data', paper_title, 'poster.png')
        qa_path = os.path.join('Paper2Poster-data', paper_title, 'o3_qa.json')

        qa_dict = json.loads(qa)
        with open(qa_path, 'w') as f:
            json.dump(qa_dict, f, indent=4)
        print(f"Saved QA for {paper_title} into {qa_path}")

        try:
            subprocess.run(['wget', paper_url, '-O', paper_output_path], check=True)
            subprocess.run(['wget', poster_url, '-O', poster_output_path], check=True)
            print(f"Downloaded {poster_url} into {poster_output_path}")
            print(f"Downloaded {paper_url} into {paper_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {paper_url} or {poster_url}: {e}")

    generate_meta_json('Paper2Poster-data')