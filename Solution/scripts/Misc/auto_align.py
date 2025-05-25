import pandas as pd
import nltk
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def penn_to_simple(pos):
    if pos.startswith('N'): return 'noun'
    if pos.startswith('V'): return 'verb'
    if pos.startswith('J'): return 'adj'
    if pos.startswith('R'): return 'adv'
    return pos.lower()

def custom_tokenize(text):
    return re.findall(r'<extra_id_\d+>|[\w\']+|[.,!?;:]', text)

df = pd.read_csv('./data/final_training_data_balanced.csv')

for i, row in df[df['swap_needed'] == 1].iterrows():
    sent = str(row["input_sentence"])
    gold_outdated = [w.strip().lower() for w in str(row.get("outdated_word(s)", "")).split(",")]
    words = custom_tokenize(sent)
    pos_tags = nltk.pos_tag(words)
    gold_pos_new = []
    for gold_word in gold_outdated:
        found_pos = ''
        for word, pos in pos_tags:
            if word.lower().strip(".,!?") == gold_word:
                found_pos = penn_to_simple(pos)
                break
        gold_pos_new.append(found_pos)
    # Join to single string for CSV cell
    df.at[i, "outdated_word(s)_pos"] = ",".join(gold_pos_new)

# Save to new file
df.to_csv('./data/final_training_data_balanced_nltkpos.csv', index=False)
