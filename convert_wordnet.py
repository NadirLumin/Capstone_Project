import pandas as pd
import os

def parse_wordnet_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.strip().split(' ')
            if len(parts) > 1:  # Ensure there are enough parts
                synset_id = parts[0]
                definition = ' '.join(parts[1:])
                data.append({'Synset ID': synset_id, 'Definition': definition})
    
    return data

def parse_index_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    index_data = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.strip().split(' ')
            if len(parts) >= 2:  # Ensure there are at least two parts
                word = parts[0]
                synset_id = parts[1]
                index_data.append({'Word': word, 'Synset ID': synset_id})
    
    return index_data

# Specify the file paths for data files
data_files = {
    'Noun': '/Users/kingcarlos/Downloads/dict/data.noun',
    'Verb': '/Users/kingcarlos/Downloads/dict/data.verb',
    'Adjective': '/Users/kingcarlos/Downloads/dict/data.adj',
    'Adverb': '/Users/kingcarlos/Downloads/dict/data.adv',
}

# Specify the file paths for index files
index_files = {
    'Noun': '/Users/kingcarlos/Downloads/dict/index.noun',
    'Verb': '/Users/kingcarlos/Downloads/dict/index.verb',
    'Adjective': '/Users/kingcarlos/Downloads/dict/index.adj',
    'Adverb': '/Users/kingcarlos/Downloads/dict/index.adv',
}

# Combine data into a DataFrame
all_data = []

# Parse data files
for pos, filepath in data_files.items():
    wordnet_data = parse_wordnet_file(filepath)
    for entry in wordnet_data:
        entry['Part of Speech'] = pos
        all_data.append(entry)

# Parse index files
for pos, filepath in index_files.items():
    index_data = parse_index_file(filepath)
    for entry in index_data:
        # Append part of speech for index entries as well
        entry['Part of Speech'] = pos
        all_data.append(entry)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
output_path = '/Users/kingcarlos/DIVINE_LUMINARY/UCSD_course/UCSD_SB_Machine_Learning_Course/Capstone_project_datasets/wordnet_data/wordnet_data.csv'
df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
