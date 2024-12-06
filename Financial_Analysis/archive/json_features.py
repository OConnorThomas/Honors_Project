import json
import os
import csv
from tqdm import tqdm
from collections import defaultdict

def process_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError:
        return f"Error decoding JSON in file: {file_path}", None

    # Dictionary to store feature counts
    feature_count = defaultdict(int)

    # Iterate through all categories and items
    for category in ['bs', 'ic', 'cf']:
        for item in data['data'].get(category, []):
            concept = item['concept']
            feature_count[concept] += 1
    
    return None, feature_count

def main_prog():
    # Parent directory containing subfolders
    parent_dir = 'data/'

    # Output directory for the global CSV file
    output_dir = 'clean_data/'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a global dictionary to store cumulative feature counts across all subdirectories
    global_feature_count = defaultdict(int)

    # Iterate through each subfolder
    subdirectories = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    bar = tqdm(total=len(subdirectories) + 1, desc='Generating')

    for subdir, _, files in os.walk(parent_dir):
        bar2 = tqdm(total=len(files), desc='Processing', leave=False)

        # Process each JSON file in the subfolder
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(subdir, file_name)
                status, feature_count = process_json_file(file_path)
                if status is None and feature_count:
                    for concept, count in feature_count.items():
                        global_feature_count[concept] += count

                bar2.update(1)

        bar.update(1)

    # Write the global results to a single CSV file
    global_csv_file_path = os.path.join(output_dir, 'global_feature_counts.csv')
    with open(global_csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['feature', 'count'])
        for concept, count in global_feature_count.items():
            writer.writerow([concept, count])

if __name__ == '__main__':
    main_prog()
