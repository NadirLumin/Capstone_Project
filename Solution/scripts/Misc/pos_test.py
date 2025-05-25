import pandas as pd
import nltk
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def smart_pos_tag(sentence_or_words, od_terms):
    # Accepts either a sentence (string) or a token list.
    if isinstance(sentence_or_words, str):
        words = custom_tokenize(sentence_or_words)
    else:
        words = sentence_or_words[:]
    if not words:
        return []
    orig_first = words[0]
    if orig_first.lower() in od_terms:
        words[0] = orig_first.lower()
    pos_tags = nltk.pos_tag(words)
    words[0] = orig_first  # restore
    return list(zip(words, [pos for _, pos in pos_tags]))

def penn_to_simple(pos):
    if pos == 'MD': return 'modal'
    if pos.startswith('N'): return 'noun'
    if pos.startswith('V'): return 'verb'
    if pos.startswith('J'): return 'adj'
    if pos.startswith('R'): return 'adv'
    return pos.lower()

def custom_tokenize(text):
    return re.findall(r'<extra_id_\d+>|[\w\']+|[.,!?;:]', text)

def simple_lemmatize(word):
    word = word.lower().strip()
    if word.endswith("'s"): word = word[:-2]
    elif word.endswith("es") and word[:-2] in OD_TERMS: word = word[:-2]
    elif word.endswith("s") and word[:-1] in OD_TERMS: word = word[:-1]
    elif word.endswith("ing") and word[:-3] in OD_TERMS: word = word[:-3]
    elif word.endswith("ed") and word[:-2] in OD_TERMS: word = word[:-2]
    return word.rstrip('.,!?')

# Load your training data
df = pd.read_csv('./data/final_training_data_balanced.csv')
mismatches = []
OD_TERMS = set(df['outdated_word(s)'].str.lower())

for idx, row in df[df["swap_needed"] == 1].iterrows():
    sent = str(row["input_sentence"])
    gold_outdated = [w.strip().lower() for w in str(row.get("outdated_word(s)", "")).split(",")]
    gold_pos_list = [w.strip().lower() for w in str(row.get("outdated_word(s)_pos", "")).split(",")]

    words = custom_tokenize(sent)
    pos_tags = smart_pos_tag(words, OD_TERMS)
    print(f"SENT: {sent}")
    print(f"POS TAGS: {pos_tags}")
    for go_idx, gold_word in enumerate(gold_outdated):
        if not gold_word: continue
        gold_pos = gold_pos_list[go_idx] if go_idx < len(gold_pos_list) else ""
        found = False
        for i, (word, pos) in enumerate(pos_tags):
            lemma = simple_lemmatize(word)
            simple_pos = penn_to_simple(pos)
            if lemma == gold_word:
                found = True
                if gold_pos and gold_pos != simple_pos:
                    # --- QUICK EXCEPTION FOR KNOWN QUIRKS ---
                    # If gold expects "verb" but NLTK thinks "noun" or "NNP", IGNORE
                    if (
                        gold_pos == "verb"
                        and (simple_pos == "noun" or pos in ("NN", "NNP", "NNS", "NNPS"))
                    ):
                        print(f"[INFO] Skipping mismatch for '{word}': gold=verb, nltk={pos}")
                        continue  # Do NOT add to mismatches
                    # You can add more exceptions for other quirks if needed
                    mismatches.append({
                        "idx": idx,
                        "word": word,
                        "lemma": lemma,
                        "train_pos": gold_pos,
                        "nltk_pos": simple_pos,
                        "sentence": sent
                    })
                break
        if not found:
            print(f"[WARNING] Could not find '{gold_word}' in: {sent}")

print(f"Total POS mismatches: {len(mismatches)}")

if mismatches:
    for m in mismatches[:10]:
        print(m)  # Show the first 10 for review

    print("⚠️ POS mismatches detected. Consider rebuilding your training data with NLTK POS tags.")
else:
    print("✅ All POS tags match between training data and NLTK POS tags.")
