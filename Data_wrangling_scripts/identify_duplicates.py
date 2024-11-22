import pandas as pd

# Load the cleaned WordNet data
file_path = "/Users/kingcarlos/DIVINE_LUMINARY/UCSD_Course/Capstone_Project/Project_datasets/wordnet_data/cleaned_wordnet_data.csv"
df = pd.read_csv(file_path, dtype=str)  # Ensure all columns are treated as strings

# Set options to prevent truncation
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Identify exact duplicates (match all columns)
duplicates = df[df.duplicated(keep=False)]

if not duplicates.empty:
    print("Duplicate Rows Identified (Exact Matches):\n")
    for index, row in duplicates.iterrows():
        print(f"Row {index}:")
        print(row.to_string(), end="\n\n")  # Print each row with full details
else:
    print("No duplicate rows found.")
