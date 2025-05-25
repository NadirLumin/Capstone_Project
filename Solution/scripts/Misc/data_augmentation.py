import pandas as pd

# Path to your file (update as needed)
infile = "data/final_training_data_balanced_nltkpos.csv"
outfile = "data/final_training_data_balanced_nltkpos_augmented.csv"

# Load your data
df = pd.read_csv(infile)

# Define new examples
new_examples = [
    {
        "input_sentence": "The project <extra_id_0> will <extra_id_1> launch soon.",
        "output_sentence": "The project <extra_id_0> shall <extra_id_1> launch soon.",
        "swap_needed": 1,
        "outdated_word(s)": "will",
        "exuberant_synonym(s)": "shall",
        "outdated_word(s)_pos": "modal"
    },
    {
        "input_sentence": "We <extra_id_0> will <extra_id_1> succeed with effort.",
        "output_sentence": "We <extra_id_0> shall <extra_id_1> succeed with effort.",
        "swap_needed": 1,
        "outdated_word(s)": "will",
        "exuberant_synonym(s)": "shall",
        "outdated_word(s)_pos": "modal"
    },
    {
        "input_sentence": "She <extra_id_0> will <extra_id_1> complete the task.",
        "output_sentence": "She <extra_id_0> shall <extra_id_1> complete the task.",
        "swap_needed": 1,
        "outdated_word(s)": "will",
        "exuberant_synonym(s)": "shall",
        "outdated_word(s)_pos": "modal"
    }
]

# Add missing columns to new examples if needed
for ex in new_examples:
    for col in df.columns:
        if col not in ex:
            ex[col] = ""

# Append new examples
df = pd.concat([df, pd.DataFrame(new_examples)], ignore_index=True)

# Drop duplicates (based on all columns)
df = df.drop_duplicates()

# Save to a new file
df.to_csv(outfile, index=False)
print(f"✅ Data augmented and deduplicated! Saved to: {outfile}")
