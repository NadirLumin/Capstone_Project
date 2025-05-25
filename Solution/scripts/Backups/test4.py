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
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

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
special_tokens_dict = {"additional_special_tokens": ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>"]}
tok.add_special_tokens(special_tokens_dict)
model = AutoModelForTokenClassification.from_pretrained(
    "prajjwal1/bert-mini", num_labels=len(LABEL2ID)
)
model.resize_token_embeddings(len(tok))


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
    if pos == 'MD': return 'modal'
    if pos.startswith('N'): return 'noun'
    if pos.startswith('V'): return 'verb'
    if pos.startswith('J'): return 'adj'
    if pos.startswith('R'): return 'adv'
    return pos.lower()

def custom_tokenize(text):
    # This treats <extra_id_0>, <extra_id_1> etc. as single tokens
    return re.findall(r'<extra_id_\d+>|[\w\']+|[.,!?;:]', text)

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

def normalize_extra_ids(text):
    return re.sub(r"<\s*extra_id_(\d+)\s*>", r"<extra_id_\1>", text)

def postprocess_spacing(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text)  # Ensure a space after punctuation if not followed by space
    text = re.sub(r"\s+<", " <", text)
    text = re.sub(r">\s+", "> ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)  # Remove any space before period at end
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

def match_outdated_term(word, pos, allow_soft_verb=True):
    base = simple_lemmatize(word)
    pos_simple = penn_to_simple(pos)
    print(f"Checking term: '{word}' (lemmatized: '{base}') POS: '{pos}' (simplified: '{pos_simple}')")
    if (base, pos_simple) in OD_TERMS_POS:
        print(f"--> MATCHED in OD_TERMS_POS: ({base}, {pos_simple})")
        return (base, pos_simple)
    if allow_soft_verb:
        pos_alternatives = {"noun", "verb"}
        all_pos = {p for (t, p) in OD_TERMS_POS if t == base}
        if len(all_pos & pos_alternatives) == 2:
            alt_pos = "verb" if pos_simple == "noun" else "noun"
            if (base, alt_pos) in OD_TERMS_POS:
                print(f"--> SOFT MATCH in OD_TERMS_POS: ({base}, {alt_pos})")
                return (base, alt_pos)
    print(f"--> NO MATCH for: ({base}, {pos_simple})")
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
    pos_tags = smart_pos_tag(words, OD_TERMS)
    tags = []
    for i, w in enumerate(words):
        match = match_outdated_term(w, pos_tags[i][1], allow_soft_verb=True)
        tags.append("OUTDATED" if match else "O")
    return tags

def clean_input_sentence(s):
    s = re.sub(r"(?<=\w)(<extra_id_\d+>)", r" \1", s)
    s = re.sub(r"(<extra_id_\d+>)(?=\w)", r"\1 ", s)
    s = re.sub(r"<unk>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*(<extra_id_\d+>)\s*", r" \1 ", s)  # Fix for any extra_id
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+\.", ".", s)  # Remove any space before period at end
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)  # Remove space before any major punctuation
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

def make_tagged_pairs(df):
    sents, tags = [], []
    for sent in df["input_sentence"]:
        words = custom_tokenize(sent)
        pos_tags = smart_pos_tag(words, OD_TERMS)
        tag_seq = ["O"] * len(words)
        for idx, (w, pos) in enumerate(pos_tags):
            match = match_outdated_term(w, pos, allow_soft_verb=True)
            if match:
                tag_seq[idx] = "OUTDATED"
        # Debug print here
        print("Input:", words)
        print("Tags :", tag_seq)
        for w, t in zip(words, tag_seq):
            print(f"{w:20} {t}")
        print("-" * 40)
        sents.append(words)
        tags.append(tag_seq)
    return sents, tags

# Example test of mapping logic:
test_word = "manage"
test_pos = "VB"
print("\n==== POS/Synonym Mapping Sanity Check ====")
base = simple_lemmatize(test_word)
pos_simple = penn_to_simple(test_pos)
print("simple_lemmatize:", base)
print("penn_to_simple:", pos_simple)
match = match_outdated_term(test_word, test_pos)
print("match_outdated_term:", match)
print("If match is not as expected, debug mapping or POS logic!")
print("==========================================\n")

def swap_outdated(sentence, model, return_indices=False, adversarial_spans=None):
    tokens = custom_tokenize(sentence)
    pos_tags = smart_pos_tag(tokens, OD_TERMS)
    words = tokens.copy()
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

    # --- GET LOGITS and FIX SHAPE ---
    logits = model(**enc).logits
    if logits.dim() == 3 and logits.shape[0] == 1:
        logits = logits[0]  # [seq_len, num_labels]

    print("logits.shape:", logits.shape)
    print("input_ids.shape:", enc["input_ids"].shape)
    print("word_ids:", word_ids)

    # --- DEBUG: PRINT TOKEN/LOGIT INFO ---
    input_ids = enc["input_ids"][0] if enc["input_ids"].ndim == 2 else enc["input_ids"]
    num_tokens = min(len(input_ids), logits.shape[0])
    for i in range(num_tokens):
        word_id = word_ids[i]
        if word_id is None:
            continue
        print(f"Token: {tok.convert_ids_to_tokens([input_ids[i]])}, WordID: {word_id}, Logits: {logits[i].detach().cpu().numpy()}, Softmax: {torch.softmax(logits[i], dim=0).detach().cpu().numpy()}")

    # --- PREDICTIONS ---
    preds = logits.argmax(dim=-1).cpu().tolist()
    print("Predicted tag ids:", preds)
    print("Decoded tags:", [ID2LABEL[p] for p in preds])
    print("Word IDs:", word_ids)
    
    for i, (wid, pred) in enumerate(zip(word_ids, preds)):
        if wid is not None and pred == LABEL2ID["OUTDATED"]:
            print(f">>> OUTDATED prediction: word idx={wid}, token={tok.convert_ids_to_tokens([input_ids[i]])}, word={words[wid]}")

    # Find OUTDATED tokens
    target_word_indices = set()
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if preds[tok_idx] == LABEL2ID["OUTDATED"]:
            target_word_indices.add(wid)
    for idx in target_word_indices:
        term = words[idx]
        pos_tag = pos_tags[idx][1] # Penn POS tag
        simple_pos = penn_to_simple(pos_tag)
        base_term = simple_lemmatize(term)
        # Try strict POS match first
        key = (base_term, simple_pos)
        synonym = hardcoded_synonym_map.get(key)

        # If no synonym, try "soft" match (noun<->verb)
        if synonym is None:
            # Try the alternative pos if present
            alt_pos = "verb" if simple_pos == "noun" else "noun"
            alt_key = (base_term, alt_pos)
            if alt_key in hardcoded_synonym_map:
                print(f"[SOFT-MATCH] Replacing '{term}' (POS: {simple_pos}) with alt POS '{alt_pos}'")
                synonym = hardcoded_synonym_map[alt_key]
        if synonym is None:
            print(f"❗ [WARN] No synonym for '{base_term}' with POS '{simple_pos}'. Leaving as is.")
            continue
        
        # Inflect plural/case as needed:
        is_plural = pos_tag in ['NNS', 'NNPS'] or (term.endswith('s') and not term.endswith('ss'))
        if is_plural and not p.singular_noun(synonym):
            synonym = p.plural(synonym)
        if term[0].isupper():
            synonym = synonym.capitalize()
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
        enc = self.tok(words, is_split_into_words=True, truncation=True, padding='max_length', max_length=self.max_len, return_offsets_mapping=True)
        labels = np.full(len(enc["input_ids"]), LABEL2ID["O"])
        last_word_id = None
        for i, word_id in enumerate(enc.word_ids()):
            if word_id is None or word_id == last_word_id:
                continue
            labels[i] = LABEL2ID[tags[word_id]]
            last_word_id = word_id
            
        # DEBUG: Print tokenization/label alignment for first few samples
        if idx < 2: # For the first few samples
            print("\n--- TOKEN DEBUG ---")
            print("Sample idx:", idx)
            print("Words:", words)
            print("Tags :", tags)
            tokens = self.tok.tokenize(" ".join(words))
            print("Tokens:", tokens)
            print("Word IDs:", enc.word_ids())
            print("Labels :", labels.tolist())
            # Nice formatted print for clarity
            for w, t in zip(words, tags):
                print(f"{w:20} {t}")
            print("-------------------\n")
            
        if idx == 0:  # or whatever debug range you want
            print("Words:", words)
            print("Tags :", tags)
            print("Tokens:", self.tok.tokenize(" ".join(words)))
            print("Word IDs:", enc.word_ids())
            print("Assigned labels:", labels.tolist())  # <--- just this!
            for i, (w, t, wid, lab) in enumerate(zip(words, tags, enc.word_ids(), labels.tolist())):
                print(f"Word: {w:12} Tag: {t:8} WordID: {wid} Label: {lab}")

        return {**{k: torch.tensor(v) for k, v in enc.items()},
                "labels": torch.tensor(labels)}
    def __len__(self):
        return len(self.word_lists)

def train_one_model(X_tr, y_tr, X_val, y_val, tr_output_sents, val_output_sents, round_num, df_val, adversarial_indices=None):
    start_time = time.time()
    model = AutoModelForTokenClassification.from_pretrained("prajjwal1/bert-mini", num_labels=len(LABEL2ID))
    model.resize_token_embeddings(len(tok))
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
            
            # Count OUTDATED tags in current batch
            num_outdated = int((labels == LABEL2ID["OUTDATED"]).sum().item())
            print(f"Batch {batch_idx // BATCH_SIZE}: OUTDATED tags in batch: {num_outdated}")
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
        
        # =================== Log VAL predictions ===================
        with open(csv_pred_log, "a", newline='') as f:
            writer = csv.writer(f)
            for idx, (input_words, ref) in enumerate(zip(X_val, val_output_sents)):
                sent = normalize_extra_ids(" ".join(input_words))
                sent = postprocess_spacing(sent)
                pred, indices = swap_outdated(sent, model, return_indices=True)
                print(f"Input: {' '.join(input_words)}")
                print(f"OUTDATED gold idx: {[i for i, t in enumerate(y_val[idx]) if t == 'OUTDATED']}")
                print(f"OUTDATED pred idx: {sorted(indices)}")
                print("-" * 40)
                pred, _ = swap_outdated(sent, model, return_indices=True)
                pred = normalize_extra_ids(pred)
                pred = postprocess_spacing(pred)
                clean_pred = postprocess_spacing(pred).strip()
                clean_ref = postprocess_spacing(ref).strip()
                if strip_extra_ids(clean_pred) == strip_extra_ids(clean_ref):
                    epoch_correct += 1
                epoch_total += 1
                clean_pred = postprocess_spacing(pred).strip()
                clean_ref = postprocess_spacing(ref).strip()
                pred_clean = strip_extra_ids(clean_pred)
                ref_clean = strip_extra_ids(clean_ref)
                gold_spans = extract_all_spans(ref_clean)
                pred_spans = extract_all_spans(pred_clean)
                swap_needed = df_val.iloc[idx]['swap_needed'] if 'swap_needed' in df_val.columns else None

                if len(gold_spans) == 0 and len(pred_spans) == 0:
                    print(f"[DEBUG: both empty] idx={idx} input={sent}")
                    print(f"  gold: {gold_spans}, pred: {pred_spans}")
                    print(f"  swap_needed: {swap_needed}")
                    print(f"  gold_ref: '{ref_clean}' pred: '{clean_pred}'")
                gold_span_str = ",".join(gold_spans)
                pred_span_str = ",".join(pred_spans)
                span_match = "ok" if gold_spans == pred_spans else "span_mismatch"
                tokens = tok.tokenize(pred)
                tokenization_issue = ""
                if "[UNK]" in tokens or "<unk>" in pred:
                    tokenization_issue = "tokenization_warning"
                
                writer.writerow([
                    "val", # Split
                    epoch,
                    idx,
                    sent,
                    clean_pred,
                    clean_ref,
                    span_match,
                    tokenization_issue,
                    gold_span_str,
                    pred_span_str
                ])
                
        # --- After validation phase for each epoch, print tag histogram ---

        val_tag_hist = Counter()
        for input_words in X_val:
            sent = normalize_extra_ids(" ".join(input_words))
            _, indices = swap_outdated(sent, model, return_indices=True)
            val_tag_hist.update(indices)
        print(f"[Epoch {epoch}] Val OUTDATED tag indices histogram:", val_tag_hist)

        # =================== Log TRAIN predictions (optional, can be slow) ===================
        if epoch == EPOCHS:
            with open(csv_pred_log, "a", newline='') as f:
                writer = csv.writer(f)
                for idx, (input_words, ref) in enumerate(zip(X_tr, tr_output_sents)):
                    sent = normalize_extra_ids(" ".join(input_words))
                    sent = postprocess_spacing(sent)
                    pred, _ = swap_outdated(sent, model, return_indices=True)
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
                        "train", # split
                        epoch,
                        idx,
                        sent,
                        clean_pred,
                        clean_ref,
                        span_match,
                        tokenization_issue,
                        gold_span_str,
                        pred_span_str
                    ])

        # Optionally print epoch accuracy:
        epoch_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        print(f"Epoch {epoch} Validation Accuracy: {epoch_accuracy:.3f}")
        val_correct = epoch_correct # Store last epoch's accuracy
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
    df = pd.read_csv("./data/final_training_data_balanced.csv")
    skf = StratifiedKFold(n_splits=NUM_ROUNDS, shuffle=True, random_state=42)
    metrics_per_round = []

    for round_num, (train_idx, test_idx) in enumerate(skf.split(df, df["swap_needed"]), 1):
        print(f"\n========== ROUND {round_num} ==========")
        df_trainval = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Overwrite preds_log.csv at the start
        with open(csv_pred_log, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
        # --- TRAIN/VAL SPLIT and PREPARE DATA ---
        df_train, df_val = train_test_split(
            df_trainval, test_size=0.2, random_state=42, stratify=df_trainval["swap_needed"]
        )

        # Prepare word/tag lists for your dataset class
        X_tr, y_tr = make_tagged_pairs(df_train)
        X_val, y_val = make_tagged_pairs(df_val)
        tr_output_sents = df_train["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing).tolist()
        val_output_sents = df_val["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing).tolist()

        ### === OVERFIT ON A SINGLE BATCH ===
        one_batch_X = X_tr[:BATCH_SIZE]
        one_batch_y = y_tr[:BATCH_SIZE]
        
        print("Overfit batch tokenization check:")
        for words in one_batch_X:
            print(tok.tokenize(" ".join(words)))

        one_batch_out = tr_output_sents[:BATCH_SIZE]

        print("\n========== OVERFIT TEST (ONE BATCH) ==========")
        overfit_result, overfit_model = train_one_model(
            one_batch_X, one_batch_y, one_batch_X, one_batch_y, one_batch_out, one_batch_out, 0, pd.DataFrame()
        )
        print("Overfit test result:", overfit_result)
        print("==============================================\n")
        ### ===========================================

    print(f"Train/Val set: {len(df_trainval)} rows, Test set: {len(df_test)} rows")
    df_test = df_test.drop_duplicates(subset=["input_sentence"])
    print(f"Train/Val set: {len(df_trainval)} rows, Test set: {len(df_test)} rows")
    print("Unique test set input sentences:", len(set(df_test["input_sentence"])))
    print("Total test rows:", len(df_test))
    print(df_test["input_sentence"].value_counts())

    # Build adversarial mapping for all rows (if needed)
    adversarial_indices = {}
    for idx, row in df.iterrows():
        if str(row.get("adversarial", "")).strip().lower() == "true":
            spans = [span.strip() for span in str(row.get("input_span", "")).split(",") if span.strip()]
            adversarial_indices[idx] = set(spans)

    print(f"Number of swap-needed (OUTDATED) sentences: {len(df[df['swap_needed']==1])}")
    print(f"Unique swap-needed sentences: {df[df['swap_needed']==1]['input_sentence'].nunique()}")

    # Final test set prep
    df_test["input_sentence"] = df_test["input_sentence"].astype(str).apply(clean_input_sentence)
    df_test["output_sentence"] = df_test["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing)
    test_word_lists = make_tagged_pairs(df_test)[0]
    test_refs = df_test["output_sentence"].tolist()

    test_adversarial_indices = {}
    for idx, row in df_test.iterrows():
        if str(row.get("adversarial", "")).strip().lower() == "true":
            spans = [span.strip() for span in str(row.get("input_span", "")).split(",") if span.strip()]
            test_adversarial_indices[idx] = set(spans)

    correct_preds = 0
    total_preds = 0
    swap_correct = 0
    no_swap_correct = 0

    # --- TRAIN/VAL SPLIT and PREPARE DATA ---
    df_train, df_val = train_test_split(
        df_trainval, test_size=0.2, random_state=42, stratify=df_trainval["swap_needed"]
    )

    # Prepare word/tag lists for your dataset class
    X_tr, y_tr = make_tagged_pairs(df_train)
    # Count OUTDATED tags in the training set
    outdated_count = sum([tag_seq.count("OUTDATED") for tag_seq in y_tr])
    print(f"Number of OUTDATED tags in training data: {outdated_count}")
    if outdated_count == 0:
        print("❗ [WARNING] No OUTDATED tags found in training data! Check mapping logic!")
    X_val, y_val = make_tagged_pairs(df_val)
    tr_output_sents = df_train["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing).tolist()
    val_output_sents = df_val["output_sentence"].astype(str).apply(normalize_extra_ids).apply(postprocess_spacing).tolist()

    result, trained_model = train_one_model(
        X_tr, y_tr, X_val, y_val, tr_output_sents, val_output_sents, round_num=round_num, df_val=df_val
    )
    
    # Do your test logging and evaluation
    metrics_per_round.append(result)
    print(f"Round {round_num} done.\n")
    print("===== ROUND METRICS =====")
    for i, m in enumerate(metrics_per_round, 1):
        print(f"Round {i}: {m}")

    # Log test set predictions with adversarial protection
    with open(csv_pred_log, "a", newline='') as f:
        writer = csv.writer(f)
        for idx, (input_words, ref) in enumerate(zip(test_word_lists, test_refs)):
            sent = normalize_extra_ids(" ".join(input_words))
            sent = postprocess_spacing(sent)
            adversarial_spans = test_adversarial_indices.get(idx, set())
            pred, _ = swap_outdated(sent, trained_model, return_indices=True, adversarial_spans=adversarial_spans)
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
                "test",  # round
                "test",  # epoch
                idx,     # idx
                sent,
                clean_pred,
                clean_ref,
                span_match,
                tokenization_issue,
                gold_span_str,
                pred_span_str
            ])
            
            swap_needed = df_test.iloc[idx]['swap_needed']
            total_preds += 1

            if swap_needed == 1:
                if gold_spans == pred_spans:
                    correct_preds += 1
                    swap_correct += 1
            elif swap_needed == 0:
                if clean_pred == clean_ref:
                    correct_preds += 1
                    no_swap_correct += 1

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    print(f"==== FINAL KEPT-OUT TEST SET RESULTS ====")
    print(f"Total predictions: {total_preds}")
    print(f"Percent correct: {accuracy * 100:.1f}%")
    print(f"Correct predictions: {correct_preds}")
    print(f"  ...of which are swap_correct: {swap_correct}")
    print(f"  ...of which are no_swap_correct: {no_swap_correct}")
    print("=========================================")
    
    print("\n===== Sample Predictions (first 5) =====")
    for i in range(min(5, len(test_word_lists))):
        sent = normalize_extra_ids(" ".join(test_word_lists[i]))
        pred, _ = swap_outdated(sent, trained_model, return_indices=True)
        gold = test_refs[i]
        print(f"[{i}] Input:    {sent}")
        print(f"    Gold:     {gold}")
        print(f"    Pred:     {pred}")
        print(f"    GoldSpan: {extract_all_spans(strip_extra_ids(gold))}")
        print(f"    PredSpan: {extract_all_spans(strip_extra_ids(pred))}")
        print("---")

    metrics_path = os.path.join(RESULTS_DIR, "final_test_metrics.txt")
    with open(metrics_path, "w") as mfile:
        mfile.write(
            f"Total: {total_preds}\n"
            f"Correct: {correct_preds}\n"
            f"Swap correct: {swap_correct}\n"
            f"No-swap correct: {no_swap_correct}\n"
            f"Accuracy: {accuracy:.3f}\n"
        )

    print(f"✅ Filtered preds_log.csv complete.")

if __name__ == "__main__":
    main()
