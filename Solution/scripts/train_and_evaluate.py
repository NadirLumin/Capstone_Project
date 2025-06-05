import csv
import inflect
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import re
import time
import torch

from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Environment variable setup
os.environ["TRANSFORMERS_NO_TF"] = "1"

# NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Inflect engine initialization
p = inflect.engine()

# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDATED_TERMS_PATH = "./data/cleaned_synonyms_data.csv"
RESULTS_DIR = "./results"
EPOCHS, BATCH_SIZE, NUM_ROUNDS, LEARNING_RATE = 20, 16, 5, 5e-5
os.makedirs(f"{RESULTS_DIR}/json", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations/plots/transformer", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/csv", exist_ok=True)
lemmatizer = WordNetLemmatizer()
LABEL2ID = {"O": 0, "OUTDATED": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
tok = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
model = AutoModelForTokenClassification.from_pretrained("prajjwal1/bert-mini", num_labels=len(LABEL2ID))

csv_pred_log = os.path.join(RESULTS_DIR, "preds_log.csv")
header = [
    "round", "epoch", "idx", "input", "prediction", "reference",
    "span_match", "tokenization_issue", "gold_span", "predicted_span"
]
if not os.path.exists(csv_pred_log):
    with open(csv_pred_log, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def _norm(term):
    term = re.sub(r"[“”‘’]", '"', term)
    return re.sub(r"\(s\)", "", term).strip().strip('"').lower()

# Load once
df_syn = pd.read_csv(OUTDATED_TERMS_PATH)
df_syn["Outdated Term"] = df_syn["Outdated Term"].astype(str).apply(_norm)
df_syn["POS"] = df_syn["POS"].astype(str).str.lower().str.strip()
df_syn["Exuberant Synonyms"] = df_syn["Exuberant Synonyms"].astype(str).apply(_norm)

# Build mapping: (term, pos) -> synonym
hardcoded_synonym_map = {
    (row["Outdated Term"], row["POS"]): row["Exuberant Synonyms"].split(",")[0].strip()
    for _, row in df_syn.iterrows()
    if row["Outdated Term"] and row["Exuberant Synonyms"] and row["POS"]
}
OD_TERMS_POS = set(hardcoded_synonym_map.keys())
OD_TERMS = set([k[0] for k in OD_TERMS_POS])

print("🔍 Allowed synonym tokens:", sorted(list(set(hardcoded_synonym_map.values()))))
print("\n🗺️ Loaded Synonym Map:")
for k, v in hardcoded_synonym_map.items():
    print(f"'{k}' -> '{v}'")
print("-" * 40)

# UTILITIES
SPAN_RE = re.compile(r"<extra_id_0>\s*(.*?)\s*<extra_id_1>", re.IGNORECASE)

# Map Penn tags to simplified (for CSV matching)
def penn_to_simple(pos):
    if pos.startswith('N'): return 'noun'
    if pos.startswith('V'): return 'verb'
    if pos.startswith('J'): return 'adj'
    if pos.startswith('R'): return 'adv'
    return pos.lower()

def custom_tokenize(text):
    # This treats <extra_id_0>, <extra_id_1> etc. as single tokens
    return re.findall(r'<extra_id_\d+>|[\w\']+|[.,!?;:]', text)

def normalize_extra_ids(text):
    return re.sub(r"<\s*extra_id_(\d+)\s*>", r"<extra_id_\1>", text)

def postprocess_spacing(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", text) # Remove space before punctuation
    text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text) # Ensure a space after punctuation if not followed by space
    text = re.sub(r"\s+<", " <", text)
    text = re.sub(r">\s+", "> ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text) # Remove any space before period at end
    text = text.strip()
    return text

def simple_lemmatize(word):
    word = word.lower().strip()
    if word.endswith("'s"): word = word[:-2]
    elif word.endswith("es") and word[:-2] in OD_TERMS: word = word[:-2]
    elif word.endswith("s") and word[:-1] in OD_TERMS: word = word[:-1]
    elif word.endswith("ing") and word[:-3] in OD_TERMS: word = word[:-3]
    elif word.endswith("ed") and word[:-2] in OD_TERMS: word = word[:-2]
    return word

def match_outdated_term(word, pos):
    base = simple_lemmatize(word)
    pos_simple = penn_to_simple(pos)
    # Direct match
    if (base, pos_simple) in OD_TERMS_POS:
        return (base, pos_simple)
    # Fallback: try all possible POS tags for this base word in your mapping
    for (t, p) in OD_TERMS_POS:
        if t == base:
            print(f"[FALLBACK] Used synonym for '{base}' with alt POS '{p}' (original POS: {pos_simple})")
            return (base, p)
    return None

def od_term_in_adversarial(row):
    # Only consider adversarial entries
    if not str(row.get("adversarial", "")).strip().lower() == "true":
        return False
    words = [w.strip(".,!?") for w in str(row["input_sentence"]).split()]
    for w in words:
        if simple_lemmatize(w) in OD_TERMS:
            return True
    return False

def tag_terms(words):
    # Requires pos_tags for each word
    pos_tags = nltk.pos_tag(words)
    tags = []
    for i, w in enumerate(words):
        match = match_outdated_term(w, pos_tags[i][1])
        tags.append("OUTDATED" if match else "O")
    return tags

def clean_input_sentence(s):
    s = re.sub(r"(?<=\w)(<extra_id_\d+>)", r" \1", s)
    s = re.sub(r"(<extra_id_\d+>)(?=\w)", r"\1 ", s)
    s = re.sub(r"<unk>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*(<extra_id_\d+>)\s*", r" \1 ", s) # Fix for any extra_id
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+\.", ".", s) # Remove any space before period at end
    return s

def strip_extra_ids(text):
    text = re.sub(r"\s*<extra_id_0>\s*", " ", text)
    text = re.sub(r"\s*<extra_id_1>\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_all_spans(text):
    # Finds all spans between matching extra_id tags: <extra_id_0>something<extra_id_1> etc.
    pattern = re.compile(r"<extra_id_\d+>\s*(.*?)\s*<extra_id_\d+>")
    return [span.strip() for span in pattern.findall(text)]

def match_capitalization(src, tgt):
    if src.isupper():
        return tgt.upper()
    elif src.istitle():
        return tgt.title()
    else:
        return tgt.lower()

def make_tagged_pairs(df):
    sents, tags, provided_pos_maps = [], [], []
    for idx, row in df.iterrows():
        sent = row["input_sentence"]
        outdated_terms = [simple_lemmatize(w.strip()) for w in str(getattr(row, 'outdated_word_s_', '')).split(",") if w.strip()]
        print(f"idx={idx} SENT={sent}")
        print(f"  OUTDATED_TERMS={outdated_terms}")
        words = custom_tokenize(sent)
        print(f"  TOKENS={words}")
        
        # Extract outdated terms and POS per row
        outdated_terms = [simple_lemmatize(w.strip()) for w in str(getattr(row, 'outdated_word_s_', '')).split(",") if w.strip()]
        outdated_pos = [p.strip().lower() for p in str(getattr(row, 'outdated_word_s__pos', '')).split(",") if p.strip()]
        provided_pos_map = dict(zip(outdated_terms, outdated_pos))

        words = custom_tokenize(sent)
        pos_tags = nltk.pos_tag(words) # Still helpful for non-outdated words!
        tag_seq = ["O"] * len(words)
        for i, (w, pos) in enumerate(pos_tags):
            base = simple_lemmatize(w)
            if base in provided_pos_map:
                correct_pos = provided_pos_map[base]
                tag_seq[i] = "OUTDATED"
                if "user" in words and "crucial" in words:
                    print(f"[DEBUG TRAIN] For sentence: '{sent}'")
                    print(f"  Words: {words}")
                    print(f"  Tag sequence: {tag_seq}")
                    print(f"  Provided POS map: {provided_pos_map}")

                print(f"[TRAIN SWAP DEBUG] (FROM DATA) idx={idx} '{w}' mapped to POS '{correct_pos}' for training.")
            else:
                # Fallback: if you want, utilize NLTK POS to tag non-outdated terms
                pass # Leave as "O"
        sents.append(words)
        tags.append(tag_seq)
        provided_pos_maps.append(provided_pos_map)
    return sents, tags, provided_pos_maps

def swap_outdated(sentence, model, provided_pos_map=None, return_indices=False, adversarial_spans=None):
    tokens = custom_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    words = tokens
    enc = tok(words, is_split_into_words=True, return_tensors="pt").to(DEVICE)
    try:
        word_ids = enc.word_ids(0)
    except Exception:
        word_ids = enc.word_ids()[0]
    if word_ids is None:
        print(f"[WARN] word_ids is None for input: {sentence}")
        joined = " ".join(words)
        joined = postprocess_spacing(joined)
        return joined if not return_indices else (joined, set())
    logits = model(**enc).logits.squeeze(0)
    preds = logits.argmax(dim=-1).cpu().tolist()
    print("  [DEBUG] Predicted logits (first 10 tokens):", logits[:10])
    print("  [DEBUG] Predicted labels:", preds[:10])
    print(f"[DEBUG] swap_outdated preds: {preds}")
    print(f"[DEBUG] word_ids: {word_ids}")
    print(f"[DEBUG] tokens: {tokens}")
    target_word_indices = set()
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if preds[tok_idx] == LABEL2ID["OUTDATED"]:
            target_word_indices.add(wid)
    if not target_word_indices:
        print(f"[SWAP DEBUG] No OUTDATED tag detected for: '{sentence}'")

    for idx in target_word_indices:
        term = words[idx]
        base_term = simple_lemmatize(term)
        print(f"\n[DEBUG] Token idx: {idx} | Original term: '{term}' | Base term: '{base_term}'")
        print(f"        Provided POS map: {provided_pos_map}")
        if provided_pos_map:
            print(f"        Available keys in POS map: {list(provided_pos_map.keys())}")
        else:
            print(f"        No POS map provided or it's empty.")
        pos_simple = None

        # Strongly prefer provided POS if possible!
        if provided_pos_map and base_term in provided_pos_map:
            pos_simple = provided_pos_map[base_term]
            print(f"        [FROM DATA] '{term}' utilizes POS '{pos_simple}' (from provided_pos_map)")
        else:
            print(f"        [WARNING] Falling back to NLTK POS for '{term}' (base '{base_term}')! Fix your data if this is unexpected.")
            pos = pos_tags[idx][1]
            pos_simple = penn_to_simple(pos)
            print(f"        [NLTK FALLBACK] '{term}' POS '{pos}' simplified to '{pos_simple}'")

        # Check for exact key in synonym map
        print(f"        Will look up: (base_term, pos_simple) = ({base_term}, {pos_simple})")
        available_pos = [k[1] for k in OD_TERMS_POS if k[0] == base_term]
        print(f"        Available POS for '{base_term}': {available_pos}")

        if (base_term, pos_simple) in hardcoded_synonym_map:
            synonym = hardcoded_synonym_map[(base_term, pos_simple)]
            print(f"        -> Synonym found: '{synonym}'")
        elif len(available_pos) == 1:
            only_pos = available_pos[0]
            synonym = hardcoded_synonym_map[(base_term, only_pos)]
            print(f"        -> Only one POS for '{base_term}'; using synonym '{synonym}' (POS '{only_pos}')")
        else:
            synonym = term
            print(f"        -> No matching POS; defaulting to original term '{term}'")

        is_plural = pos_tags[idx][1] in ['NNS', 'NNPS'] or (term.endswith('s') and not term.endswith('ss'))
        if synonym != term:
            if is_plural:
                if not p.singular_noun(synonym):
                    synonym = p.plural(synonym)
            synonym = match_capitalization(term, synonym)
            words[idx] = synonym

    joined = " ".join(words)
    joined = postprocess_spacing(joined)
    if return_indices:
        return joined, target_word_indices
    return joined
class TagDataset(Dataset):
    def __init__(self, word_lists, tag_lists, tokenizer, max_len=96):
        self.word_lists, self.tag_lists = word_lists, tag_lists
        self.tok, self.max_len = tokenizer, max_len
    def __getitem__(self, idx):
        words, tags = self.word_lists[idx], self.tag_lists[idx]
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        labels = np.full(len(enc["input_ids"]), -100) # Ignore by default!
        last_word_id = None
        for i, word_id in enumerate(enc.word_ids()):
            if word_id is None or word_id == last_word_id:
                continue # Only label first subword of each word
            labels[i] = LABEL2ID[tags[word_id]] # Assign 0 or 1
            last_word_id = word_id

        # DEBUG: Print tokenization/label alignment for first few samples
        if idx < 2:
            print("\n--- TOKEN DEBUG ---")
            print("Sample idx:", idx)
            print("Words:", words)
            print("Tags:", tags)
            tokens = self.tok.tokenize(" ".join(words))
            print("Tokens:", tokens)
            print("Word IDs:", enc.word_ids())
            print("Labels:", labels.tolist())
            print("-------------------\n")

        return {**{k: torch.tensor(v) for k, v in enc.items()},
                "labels": torch.tensor(labels)}
    def __len__(self):
        return len(self.word_lists)

def train_one_model(X_tr, y_tr, X_val, y_val, X_val_df, round_num, adversarial_indices=None):
    start_time = time.time()
    model = AutoModelForTokenClassification.from_pretrained("prajjwal1/bert-mini", num_labels=len(LABEL2ID))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_ds = TagDataset(X_tr, y_tr, tok)
    csv_pred_log = os.path.join(RESULTS_DIR, "preds_log.csv")

    losses = []
    val_correct = 0
    val_total = 0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        for batch_idx in range(0, len(train_ds), BATCH_SIZE):
            batch = [train_ds[i] for i in range(batch_idx, min(batch_idx+BATCH_SIZE, len(train_ds)))]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(DEVICE)
            labels = torch.stack([b["labels"] for b in batch]).to(DEVICE)
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Debug prints
            print(f"Epoch {epoch} Batch {batch_idx // BATCH_SIZE}: Loss={loss.item():.4f}")

        # Validation phase after each epoch
        model.eval()
        epoch_correct = 0
        epoch_total = 0
        with torch.no_grad():
            print(f"\n[VAL] Sample predictions after epoch {epoch}:")
            
            with open(csv_pred_log, "a", newline='') as f:
                writer = csv.writer(f)
                for idx, (input_words, ref, row) in enumerate(zip(X_val, y_val, X_val_df.itertuples(index=False))):
                    sent = normalize_extra_ids(" ".join(input_words))
                    sent = postprocess_spacing(sent)
                    # Attribute names are column names with non-alphanumerics replaced by underscores:
                    outdated_terms = [simple_lemmatize(w.strip()) for w in str(getattr(row, 'outdated_word_s_', '')).split(",") if w.strip()]
                    outdated_pos = [p.strip().lower() for p in str(getattr(row, 'outdated_word_s__pos', '')).split(",") if p.strip()]
                    provided_pos_map = dict(zip(outdated_terms, outdated_pos))
                    adversarial_spans = adversarial_indices.get(idx, set())
                    print(f"[VAL DEBUG] Provided POS map: {provided_pos_map}")
                    print(f"[VAL DEBUG] Sentence: {sent}")
                    pred, _ = swap_outdated(
                        sent,
                        model,
                        provided_pos_map=provided_pos_map,
                        return_indices=True,
                        adversarial_spans=adversarial_spans
                    )
                    print(f"[VAL DEBUG] Model preds: {pred}")
                    pred = normalize_extra_ids(pred)
                    pred = postprocess_spacing(pred)
                    clean_pred = postprocess_spacing(pred).strip()
                    clean_ref = postprocess_spacing(ref).strip()
                    pred_clean = strip_extra_ids(clean_pred)
                    ref_clean = strip_extra_ids(clean_ref)
                    gold_spans = extract_all_spans(ref_clean)
                    pred_spans = extract_all_spans(pred_clean)
                    gold_span_str = ",".join(gold_spans)
                    pred_span_str = ",".join(pred_spans)
                    span_match = "ok" if gold_spans == pred_spans else "span_mismatch"
                    tokens = tok.tokenize(pred)
                    tokenization_issue = ""
                    if "[UNK]" in tokens or "<unk>" in pred:
                        tokenization_issue = "tokenization_warning"
                    writer.writerow([
                        round_num, epoch, idx, sent, clean_pred, clean_ref,
                        span_match, tokenization_issue, gold_span_str, pred_span_str
                    ])
                    if gold_spans == pred_spans:
                        epoch_correct += 1
                    epoch_total += 1

        # Optionally print epoch accuracy:
        epoch_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        print(f"Epoch {epoch} Validation Accuracy: {epoch_accuracy:.3f}")
        val_correct = epoch_correct  # Store last epoch's accuracy
        val_total = epoch_total
        model.train()

    # Save model after all epochs
    model_save_path = os.path.join(RESULTS_DIR, f"model_round_{round_num}")
    model.save_pretrained(model_save_path)
    tok.save_pretrained(model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    duration = time.time() - start_time
    plt.figure()
    plt.plot(losses)
    plt.title(f"Loss Curve - Round {round_num}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{RESULTS_DIR}/visualizations/plots/transformer/loss_curve_round{round_num}.png")
    plt.close()

    round_accuracy = val_correct / val_total if val_total else 0.0
    return {"duration": duration, "accuracy": round_accuracy}, model

def has_repetition(text, proximity=3, window_size=2):
    tokens = text.lower().split()
    counts = Counter(tokens)
    for i in range(len(tokens) - window_size):
        window = tokens[i:i + window_size]
        if len(set(window)) == 1 and window[0] in counts and counts[window[0]] >= proximity:
            return True
    return False

def main():
    with open(csv_pred_log, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    df = pd.read_csv("./data/final_training_data_balanced.csv")

    # Kept out test set ONCE
    df_trainval, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["swap_needed"]
    )
    df_trainval = df_trainval.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print(f"[CHECK] df_trainval shape before eval: {df_trainval.shape}")
    print(f"[CHECK] df_test shape before eval: {df_test.shape}")
    print(f"Train/Val set: {len(df_trainval)} rows, Test set: {len(df_test)} rows")
    print(f"[DEBUG] Train/Val swap_needed counts:\n{df_trainval['swap_needed'].value_counts()}")
    print(f"[DEBUG] Test swap_needed counts:\n{df_test['swap_needed'].value_counts()}")
    print(f"[DEBUG] df_test shape: {df_test.shape}")
    print(f"[DEBUG] df shape: {df.shape}")
    print(f"[DEBUG] df_trainval.shape: {df_trainval.shape}, df_test.shape: {df_test.shape}")
    print(f"[DEBUG] df_trainval indices: {df_trainval.index[:10].tolist()}")
    print(f"[DEBUG] df_test indices: {df_test.index[:10].tolist()}")
    print("[DEBUG] First 3 rows trainval:\n", df_trainval.head(3))
    print("[DEBUG] First 3 rows test:\n", df_test.head(3))

    NUM_ROUNDS = 5
    kf = KFold(n_splits=NUM_ROUNDS, shuffle=True, random_state=42)
    X = df_trainval.reset_index(drop=True)
    for round_num, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_df = X.iloc[train_idx].reset_index(drop=True)
        X_val_df = X.iloc[val_idx].reset_index(drop=True)

    round_stats = []
    for round_num, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Round {round_num}/{NUM_ROUNDS} ===")
        X_train_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx] # Keep DataFrames for reference columns
        
        # ---- Your tagging and tag checking goes HERE ----
        train_word_lists, train_tags, train_pos_maps = make_tagged_pairs(X_train_df)
        val_word_lists, val_tags, val_pos_maps = make_tagged_pairs(X_val_df)

        # Place the train set tag check HERE
        print(f"===== Sanity check on training tags for round {round_num} =====")
        for idx, (row, tags) in enumerate(zip(X_train_df.itertuples(index=False), train_tags)):
            swap_needed = getattr(row, 'swap_needed', 0)
            if int(swap_needed) == 1 and "OUTDATED" not in tags:
                print(f"[TAG ERROR] idx={idx} swap_needed==1 but no OUTDATED in tags! Sentence: {row.input_sentence}")
            if int(swap_needed) == 0 and "OUTDATED" in tags:
                print(f"[TAG ERROR] idx={idx} swap_needed==0 but OUTDATED present in tags! Sentence: {row.input_sentence}")

        # Optional: Do the same for validation
        print(f"===== Sanity check on validation tags for round {round_num} =====")
        for idx, (row, tags) in enumerate(zip(X_val_df.itertuples(index=False), val_tags)):
            swap_needed = getattr(row, 'swap_needed', 0)
            if int(swap_needed) == 1 and "OUTDATED" not in tags:
                print(f"[VAL TAG ERROR] idx={idx} swap_needed==1 but no OUTDATED in tags! Sentence: {row.input_sentence}")
            if int(swap_needed) == 0 and "OUTDATED" in tags:
                print(f"[VAL TAG ERROR] idx={idx} swap_needed==0 but OUTDATED present in tags! Sentence: {row.input_sentence}")
                
        train_word_lists, train_tags, _ = make_tagged_pairs(X_train_df)
        val_word_lists, val_tags, _ = make_tagged_pairs(X_val_df)
        print(f"[VAL TAG CHECK] Round {round_num}:")
        for i, tags in enumerate(val_tags[:10]):  # Print only first 10 for brevity
            print(f"Example {i}: {tags}  |  OUTDATED present: {'OUTDATED' in tags}")
        val_refs = X_val_df["output_sentence"].tolist()

        # Adversarial mapping for validation
        val_adversarial_indices = {}
        for idx, row in X_val_df.iterrows():
            if str(row.get("adversarial", "")).strip().lower() == "true":
                spans = [span.strip() for span in str(row.get("input_span", "")).split(",") if span.strip()]
                val_adversarial_indices[idx] = set(spans)

        stats, _ = train_one_model(
            train_word_lists,
            train_tags,
            val_word_lists,
            val_refs,
            X_val_df,
            round_num,
            adversarial_indices=val_adversarial_indices,
        )
        round_stats.append((round_num, stats["accuracy"]))

    # After KRound rounds, before picking best round/model
    # Save per-round scores
    pd.DataFrame(round_stats, columns=["round", "val_accuracy"]).to_csv(
        os.path.join(RESULTS_DIR, "csv", "per_round_val_accuracy.csv"), index=False
    )
    with open(os.path.join(RESULTS_DIR, "json", "per_round_val_accuracy.json"), "w") as f:
        json.dump(round_stats, f)

    # Compute and save mean/stddev metrics
    accs = [a for (_, a) in round_stats]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"Mean validation accuracy: {mean_acc:.4f}")
    print(f"Std dev validation accuracy: {std_acc:.4f}")

    # Save as JSON
    metrics_path = os.path.join(RESULTS_DIR, "json", "val_accuracy_stats_summary.json")
    with open(metrics_path, "w") as jf:
        json.dump({
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "per_round": round_stats
        }, jf, indent=2)

    # ==== Pick the best model by validation accuracy ====
    best_round, best_acc = max(round_stats, key=lambda x: x[1])
    print(f"\n🏅 Best round: {best_round} (Validation accuracy: {best_acc:.4f})")
    best_model_path = os.path.join(RESULTS_DIR, f"model_round_{best_round}")
    
    # Reload best model/tokenizer
    model = AutoModelForTokenClassification.from_pretrained(best_model_path)
    tok = AutoTokenizer.from_pretrained(best_model_path)

    # Final kept-out test set diagnostics
    print(f"Number of swap-needed (OUTDATED) sentences: {len(df[df['swap_needed']==1])}")
    print(f"Unique swap-needed sentences: {df[df['swap_needed']==1]['input_sentence'].nunique()}")

    # Prepare for test set evaluation
    df_test["input_sentence"] = df_test["input_sentence"].astype(str).apply(clean_input_sentence)
    df_test["output_sentence"] = df_test["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing)
    print(f"[CHECK] df_test shape before test eval: {df_test.shape}")
    test_word_lists = make_tagged_pairs(df_test)[0]
    test_refs = df_test["output_sentence"].tolist()

    test_adversarial_indices = {}
    for idx, row in df_test.iterrows():
        if str(row.get("adversarial", "")).strip().lower() == "true":
            spans = [span.strip() for span in str(row.get("input_span", "")).split(",") if span.strip()]
            test_adversarial_indices[idx] = set(spans)

    # Only one evaluation block!
    with open(csv_pred_log, "a", newline='') as f:
        writer = csv.writer(f)
        pred_ok_list = []

        for idx, (input_words, ref) in enumerate(zip(test_word_lists, test_refs)):
            sent = normalize_extra_ids(" ".join(input_words))
            sent = postprocess_spacing(sent)
            adversarial_spans = test_adversarial_indices.get(idx, set())
            row_obj = df_test.iloc[idx]
            outdated_terms = [simple_lemmatize(w.strip()) for w in str(getattr(row_obj, 'outdated_word_s_', '')).split(",") if w.strip()]
            outdated_pos = [p.strip().lower() for p in str(getattr(row_obj, 'outdated_word_s__pos', '')).split(",") if p.strip()]
            provided_pos_map = dict(zip(outdated_terms, outdated_pos))

            pred, _ = swap_outdated(
                sent, model,
                provided_pos_map=provided_pos_map,
                return_indices=True,
                adversarial_spans=adversarial_spans
            )
            pred = normalize_extra_ids(pred)
            pred = postprocess_spacing(pred)
            clean_pred = postprocess_spacing(pred).strip()
            clean_ref = postprocess_spacing(ref).strip()
            pred_clean = strip_extra_ids(clean_pred)
            ref_clean = strip_extra_ids(clean_ref)
            gold_spans = extract_all_spans(ref_clean)
            pred_spans = extract_all_spans(pred_clean)
            ok = (gold_spans == pred_spans)
            pred_ok_list.append(ok)
            
            # ********* LOG HERE! *********
            gold_span_str = ",".join(gold_spans)
            pred_span_str = ",".join(pred_spans)
            span_match = "ok" if gold_spans == pred_spans else "span_mismatch"
            tokenization_issue = ""
            tokens = tok.tokenize(pred)
            if "[UNK]" in tokens or "<unk>" in pred:
                tokenization_issue = "tokenization_warning"

            # For TEST SET, log "test", "test" instead of round/epoch!
            writer.writerow([
                "test", "test", idx, sent, clean_pred, clean_ref,
                span_match, tokenization_issue, gold_span_str, pred_span_str
            ])

        # ---- AFTER the for loop ----
        pred_ok_mask = np.array(pred_ok_list)

        swap_needed_mask = (df_test['swap_needed'].values == 1)
        no_swap_needed_mask = (df_test['swap_needed'].values == 0)
        if 'adversarial' in df_test.columns:
            adversarial_mask = (df_test['adversarial'].astype(str).str.lower().values == "true")
        else:
            adversarial_mask = np.zeros(len(df_test), dtype=bool)

        # Final counts/accuracies (vectorized, correct)
        total = len(df_test)
        correct = pred_ok_mask.sum()
        total_swap_needed = swap_needed_mask.sum()
        correct_swap_needed = (swap_needed_mask & pred_ok_mask).sum()
        total_no_swap_needed = no_swap_needed_mask.sum()
        correct_no_swap_needed = (no_swap_needed_mask & pred_ok_mask).sum()
        total_adversarial = adversarial_mask.sum()
        correct_adversarial = (adversarial_mask & pred_ok_mask).sum()

        print(f"==== FINAL KEPT-OUT TEST SET RESULTS ====")
        print(f"Total test entries: {total}")
        print(f"Overall accuracy: {correct / total:.2%} ({correct} / {total})")
        print(f"Swap-needed accuracy: {correct_swap_needed / total_swap_needed:.2%} ({correct_swap_needed} / {total_swap_needed})")
        print(f"No-swap-needed accuracy: {correct_no_swap_needed / total_no_swap_needed:.2%} ({correct_no_swap_needed} / {total_no_swap_needed})")
        print(f"Adversarial untouched accuracy: {correct_adversarial / total_adversarial:.2%} ({correct_adversarial} / {total_adversarial})")

    metrics_path = os.path.join(RESULTS_DIR, "final_test_metrics.txt")
    with open(metrics_path, "w") as mfile:
        accuracy = correct / total if total else 0.0
        mfile.write(
            f"Total: {total}\n"
            f"Correct: {correct}\n"
            f"Swap correct: {correct_swap_needed}\n"
            f"No-swap correct: {correct_no_swap_needed}\n"
            f"Accuracy: {accuracy:.3f}\n"
        )

    print(f"✅ Filtered preds_log.csv complete.")

if __name__ == "__main__":
    main()
