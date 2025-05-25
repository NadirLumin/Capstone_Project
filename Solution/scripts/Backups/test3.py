# ===================== 🛑 Block TensorFlow Auto-Import =====================

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ======================== 🤖 Hugging Face Transformers ========================
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TopKLogitsWarper,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    StoppingCriteria,
)

# ========================= 🔧  Standard Library  =========================
import gc
import inspect
import json
import logging
import re
import string
import sys
import time
from collections import Counter

# ======================= 📦  Third‑Party Libraries  =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

# ========================= 🔨  Local Modules  ============================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))

from data_loader import load_datasets
from metrics import (
    compute_bleu,
    compute_perplexity,
    compute_rouge,
    compute_synonym_bleu,
    compute_transformation_rate,
)

# ========================== ⚙️  CONFIGURATION  ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDATED_TERMS_PATH = "./data/cleaned_synonyms_data.csv"
RESULTS_DIR = "./results"
PRED_LOG_PATH = os.path.join(RESULTS_DIR, "preds_log.txt")
with open(PRED_LOG_PATH, "w") as f:
    f.write("") # Clear previous predictions
EPOCHS, BATCH_SIZE = 10, 16
NUM_FOLDS, ACCUM_STEPS = 5, 3
LEARNING_RATE = 2e-5
frequency_proximity = 5
os.makedirs(f"{RESULTS_DIR}/json", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations/plots/transformer", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/csv", exist_ok=True)
lemmatizer = WordNetLemmatizer()

# =================== MISC ===================

class StopAfterExtraID1(StoppingCriteria):
    def __init__(self, tokenizer, target_token="<extra_id_1>"):
        self.token_id = tokenizer.convert_tokens_to_ids(target_token)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # If any generated sequence ends with <extra_id_1>, stop generation
        return (input_ids[0, -1].item() == self.token_id)

# =================== 🧠 Load Hardcoded Synonyms ===================

def _norm(term: str) -> str:
    term = re.sub(r"[“”‘’]", lambda m: '"' if m.group(0) in "“”" else "'", term)
    return re.sub(r"\(s\)", "", term).strip().strip('"').lower()

df_syn = pd.read_csv(OUTDATED_TERMS_PATH)
df_syn["Outdated Term"] = df_syn["Outdated Term"].astype(str).apply(_norm)
df_syn["Exuberant Synonyms"] = df_syn["Exuberant Synonyms"].astype(str).apply(_norm)

hardcoded_synonym_map = {
    row["Outdated Term"]: row["Exuberant Synonyms"].split(",")[0].strip()
    for _, row in df_syn.iterrows()
    if row["Outdated Term"] and row["Exuberant Synonyms"]
}

OUTDATED_TERMS = list(hardcoded_synonym_map.keys())
ALL_SYNONYM_TOKENS = set(hardcoded_synonym_map.values())
print("🔍 Allowed synonym tokens:", sorted(list(ALL_SYNONYM_TOKENS)))

# ===================== 🧠 Utility Functions =====================

SPAN_RE = re.compile(r"<extra_id_0>\s*(.*?)\s*<extra_id_1>", re.IGNORECASE)

def strip_special_tokens(text: str) -> str:
    return re.sub(r"(</s>|<pad>)+", "", text).strip()

def normalize_refs(refs, tok):
    # Convert references from string to token IDs first, then decode
    token_ids = [tok.encode(ref, add_special_tokens=False) for ref in refs]
    return [tok.decode(ref, skip_special_tokens=True) for ref in token_ids]

def get_tok(path="./t5_base_model"):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.add_special_tokens({'unk_token': '<unk>', 'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>']})
    
    # Check tokenization of special tokens
    print(tok.tokenize("<extra_id_0>"))  # Print tokenization of <extra_id_0>
    print(tok.tokenize("<extra_id_1>"))  # Print tokenization of <extra_id_1>

    return tok

def get_model(tok, path="./t5_base_model"):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
    model.resize_token_embeddings(len(tok))
    return model.to(DEVICE)

def filter_valid_synonyms(tok):
    valid = set()
    for s in ALL_SYNONYM_TOKENS:
        tokens = tok.tokenize(s)
        if len(tokens) == 1 and tok.convert_tokens_to_ids(tokens[0]) != tok.unk_token_id:
            valid.add(s)
        else:
            print(f"❌ Banned multi-token synonym: '{s}' → {tokens}")
    return valid

def apply_hardcoded_synonym(txt: str) -> str:
    def _sub(m):
        term = m.group(1).strip().lower()
        replacement = hardcoded_synonym_map.get(term, term).strip()
        
        # Always add surrounding spaces to avoid collisions
        return f" <extra_id_0> {replacement} <extra_id_1> "
    
    return SPAN_RE.sub(_sub, txt).strip()

def simple_lemmatize(word):
    word = word.lower().strip()
    if word.endswith("'s"):
        word = word[:-2]
    elif word.endswith("es") and word[:-2] in OUTDATED_TERMS:
        word = word[:-2]
    elif word.endswith("s") and word[:-1] in OUTDATED_TERMS:
        word = word[:-1]
    elif word.endswith("ing") and word[:-3] in OUTDATED_TERMS:
        word = word[:-3]
    elif word.endswith("ed") and word[:-2] in OUTDATED_TERMS:
        word = word[:-2]
    return word

def match_outdated_term(word: str) -> str:
    base = simple_lemmatize(word)
    return base if base in OUTDATED_TERMS else None

def decode_reference(tok, ids):
    ref = tok.decode(ids, skip_special_tokens=False)
    ref = normalize_spacing_around_extra_ids(ref)
    ref = re.sub(r"<unk>", "", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\s+", " ", ref).strip()
    return smart_space_fix(ref)

def faithful_decode(tok, ids):
    text = tok.decode(ids, skip_special_tokens=False)
    text = normalize_spacing_around_extra_ids(text)

    # 🔒 NEW — throw away anything **after** the first closing tag
    end = text.lower().find("<extra_id_1>")
    if end != -1:
        text = text[: end + len("<extra_id_1>")]

    text = re.sub(r"<unk>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return smart_space_fix(text)

def clean_input_sentence(s):
    s = re.sub(r"(?<=\w)(<extra_id_\d+>)", r" \1", s)
    s = re.sub(r"(<extra_id_\d+>)(?=\w)", r"\1 ", s)
    s = re.sub(r"<unk>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*(<extra_id_0>|<extra_id_1>)\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = smart_space_fix(s)
    return s

def smart_space_fix(text):
    fixes = {
        r"\bSheshall\b": "She shall",
        r"\bHewill\b": "He will",
        r"\bTheyare\b": "They are",
        r"\bThemanager\b": "The manager",
        r"\bItsupport\b": "It support",
        r"(?<!\b)(?<!\s)(shall)\b": r" shall"
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def reformat_output_sentence(y: str) -> str:
    y = smart_space_fix(y)  # 👈 Fix fused words before continuing

    # Ensure space *after* <extra_id_0> and *before* <extra_id_1>
    y = re.sub(r"<extra_id_0>\s*", "<extra_id_0> ", y)
    y = re.sub(r"\s*<extra_id_1>", " <extra_id_1>", y)

    # Clean up extra spacing and unknown tokens
    y = re.sub(r"<unk>", "", y, flags=re.IGNORECASE)
    y = re.sub(r"\s+", " ", y).strip()

    return y

def normalize_spacing_around_extra_ids(txt: str) -> str:
    # Add space before and after the token if it's stuck to a word
    txt = re.sub(r"(?<=\w)(<extra_id_\d+>)", r" \1", txt)  # word<extra_id>
    txt = re.sub(r"(<extra_id_\d+>)(?=\w)", r"\1 ", txt)  # <extra_id>word
    # Standard spacing cleanup
    txt = re.sub(r"\s*(<extra_id_\d+>)\s*", r" \1 ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def exact_span_match(pred, gold_span):
    match = re.search(r"<extra_id_0>\s*(.*?)\s*<extra_id_1>", pred)
    if match:
        return match.group(1).strip().lower() == gold_span.strip().lower()
    return False

def normalize_tokens(toks):
    return set(lemmatizer.lemmatize(t.lower().strip("’s")) for t in toks)

def contains_synonym(pred):
    return any(s in normalize_tokens(pred.split()) for s in ALL_SYNONYM_TOKENS)

def detect_hallucination(pred):
    return any(t in pred.lower() for t in OUTDATED_TERMS)

def safe_decode_input(tok, ids):
    ids = ids[ids != tok.pad_token_id]
    txt = tok.decode(ids, skip_special_tokens=False)

    txt = re.sub(r"</?s>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*/s\b", "", txt)
    txt = re.sub(r"\s*<\s*extra_id_(\d+)\s*>", r" <extra_id_\1> ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    # Capitalize the word immediately after <extra_id_0> if it's the first real token
    match = re.match(r"^<extra_id_0>\s+(\w+)", txt)
    if match:
        first_word = match.group(1).capitalize()
        txt = re.sub(rf"^<extra_id_0>\s+{re.escape(match.group(1))}", f"<extra_id_0> {first_word}", txt)
    elif txt:
        txt = txt[0].upper() + txt[1:]

    return txt

def make_tagged_pairs(df):
    sents, tags = [], []
    for sent in df["input_sentence"]:
        words = sent.split()
        # mark first outdated term (simple lemma check you already wrote)
        tag_seq = ["O"] * len(words)
        for idx, w in enumerate(words):
            if match_outdated_term(w):          # your helper
                tag_seq[idx] = "OUTDATED"
                break                            # only one tag
        sents.append(words)
        tags.append(tag_seq)
    return sents, tags

def swap_outdated(sentence: str) -> str:
    words = sentence.split()
    enc = tok(words, is_split_into_words=True, return_tensors="pt").to(DEVICE)
    logits = model(**enc).logits.squeeze(0)
    preds  = logits.argmax(dim=-1).cpu().tolist()

    # resolve to *first* OUTDATED label on a whole‑word boundary
    word_ids = enc.word_ids()[0]
    target_word_idx = None
    for tok_idx, wid in enumerate(word_ids):
        if wid is None: continue
        if preds[tok_idx] == LABEL2ID["OUTDATED"]:
            target_word_idx = wid
            break
    if target_word_idx is None:
        return sentence                          # fallback: nothing tagged

    term = words[target_word_idx]
    repl = hardcoded_synonym_map.get(simple_lemmatize(term), term)
    # preserve initial cap if required
    if term[0].isupper(): repl = repl.capitalize()
    words[target_word_idx] = repl
    return " ".join(words)

# ===================== 🔠 Dataset =====================

class SentenceDataset(Dataset):
    def __init__(self, X, y, tok, max_len=70):
        self.X, self.y, self.tok, self.max_len = X, y, tok, max_len

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = clean_input_sentence(self.X[idx])
        y = self.y[idx]

        if isinstance(y, torch.Tensor):
            y = y.squeeze().cpu().numpy()
            y = ''.join([chr(i) for i in y if i != 0])
        y = y.strip()

        if not re.search(r"\b(extra_id_0|<extra_id_0>)\b", y) or not re.search(r"\b(extra_id_1|<extra_id_1>)\b", y):
            print(f"⚠️ Output missing span formatting → {y}")

        enc = self.tok(x, max_length=self.max_len, truncation=True,
                    padding="max_length", return_tensors="pt")
        dec = self.tok(y, max_length=self.max_len, truncation=True,
                    padding="max_length", return_tensors="pt")

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": dec["input_ids"].squeeze(0),
        }
        
class TagDataset(Dataset):
    def __init__(self, word_lists, tag_lists, tokenizer, max_len=96):
        self.word_lists, self.tag_lists = word_lists, tag_lists
        self.tok, self.max_len = tokenizer, max_len

    def __getitem__(self, idx):
        words, tags = self.word_lists[idx], self.tag_lists[idx]
        enc = self.tok(words,
                       is_split_into_words=True,
                       truncation=True,
                       padding='max_length',
                       max_length=self.max_len,
                       return_offsets_mapping=True)
        labels = np.full(len(enc["input_ids"]), LABEL2ID["O"])
        last_word_id = None
        for i, word_id in enumerate(enc.word_ids()):
            if word_id is None or word_id == last_word_id:
                continue                        # ignore sub‑pieces
            labels[i] = LABEL2ID[tags[word_id]]
            last_word_id = word_id
        return {**{k: torch.tensor(v) for k, v in enc.items()},
                "labels": torch.tensor(labels)}

    def __len__(self):
        return len(self.word_lists)

# ===================== 🔒 Logits Processor =====================

class SynonymConstrainedTrainer(Trainer):
    def __init__(self, *args, synonym_token_ids=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.synonym_token_ids = synonym_token_ids
        self._tokenizer = tokenizer # ✅ Store manually since base no longer does

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # Add decoder_input_ids if needed
        if "decoder_input_ids" not in inputs:
            inputs["decoder_input_ids"] = model._shift_right(labels)

        outputs = model(**inputs)
        logits = outputs.logits.clone()

        # Initialize global span
        start, end = None, None

        for i in range(logits.size(0)): # Batch dimension
            label_ids = labels[i]
            for j in range(label_ids.size(0)):
                tok_id = label_ids[j].item()
                if tok_id == self._tokenizer.convert_tokens_to_ids("<extra_id_0>"):
                    start = j + 1
                elif tok_id == self._tokenizer.convert_tokens_to_ids("<extra_id_1>") and start is not None:
                    end = j
                    break

            if start is not None and end is not None:
                full_mask = torch.full_like(logits[i, start:end], float('-inf'))
                full_mask[:, self.synonym_token_ids] = 0
                logits[i, start:end] = logits[i, start:end] + full_mask

        # 🔒 SAFETY CHECK: if span is invalid, return dummy loss
        if start is None or end is None or end <= start:
            return torch.tensor(0.0, requires_grad=True).to(logits.device)

        loss_fct = nn.CrossEntropyLoss(ignore_index=self._tokenizer.pad_token_id)

        batch_size = logits.size(0)
        loss = 0.0
        count = 0

        for i in range(batch_size):
            label_ids = labels[i]
            logit_row = logits[i]

            # Find span
            try:
                start = (label_ids == self._tokenizer.convert_tokens_to_ids("<extra_id_0>")).nonzero(as_tuple=True)[0].item() + 1
                end = (label_ids == self._tokenizer.convert_tokens_to_ids("<extra_id_1>")).nonzero(as_tuple=True)[0].item()
            except IndexError:
                continue  # Skip if no span found

            # ----- a) Inside span: only penalize if wrong synonym is chosen -----
            logits_inside = logit_row[start:end]
            labels_inside = label_ids[start:end]

            if (end > start):
                loss_inside = loss_fct(logits_inside, labels_inside)
            else:
                loss_inside = torch.tensor(0.0, device=logits.device)

            # ----- b) Outside span: penalize *any* difference (force copy) -----
            mask = torch.ones_like(label_ids, dtype=torch.bool)
            mask[start:end] = False  # Don't include inside-span tokens

            logits_outside = logit_row[mask]
            labels_outside = label_ids[mask]

            loss_outside = loss_fct(logits_outside, labels_outside)

            loss += loss_inside + loss_outside
            count += 1

        loss = loss / count if count > 0 else torch.tensor(0.0, device=logits.device)

        return (loss, outputs) if return_outputs else loss

class SynonymSlotEnforcer(LogitsProcessor):
    def __init__(self, synonym_token_ids_per_sample, tokenizer):
        self.synonym_token_ids_per_sample = synonym_token_ids_per_sample
        self.tokenizer = tokenizer
        self.extra_id_0 = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
        self.state = {}  # Track per-sample slot status

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            seq = input_ids[i]
            last_tok = seq[-1].item()
            if i not in self.state:
                self.state[i] = {"inside": False, "used": False}

            # Toggle state if <extra_id_*> seen
            if last_tok == self.extra_id_0:
                self.state[i]["inside"] = True
                self.state[i]["used"] = False
                return scores
            elif last_tok == self.extra_id_1:
                self.state[i]["inside"] = False
                return scores

            if self.state[i]["inside"]:
                if not self.state[i]["used"]:
                    # Force single allowed token
                    allowed = self.synonym_token_ids_per_sample[i]
                    mask = torch.full_like(scores[i], float('-inf'))
                    if allowed:
                        mask[allowed[0]] = 0.0
                    scores[i] = mask
                    self.state[i]["used"] = True
                else:
                    # After the synonym: force <extra_id_1> or EOS
                    mask = torch.full_like(scores[i], float('-inf'))
                    mask[self.extra_id_1] = 0.0
                    scores[i] = mask
        return scores
    
def make_logits_processor_per_sample(tok, output_spans):
    allowed_token_ids_per_sample = []

    for span in output_spans:
        tok_id = tok.convert_tokens_to_ids(span.strip())
        allowed_token_ids_per_sample.append([tok_id] if tok_id != tok.unk_token_id else [])

    return LogitsProcessorList([
        SynonymSlotEnforcer(tok, allowed_token_ids_per_sample)
    ])

# ===================== 🧪 Preview Eval =====================

def run_preview_evaluation(model, tok, X_val, y_val, use_constraints=True, epoch=None):
    logits_processor = make_logits_processor_per_sample(tok, output_spans) if use_constraints else None
    preds, refs = [], []
    halluc, missing = 0, 0
    loader = DataLoader(SentenceDataset(X_val[:BATCH_SIZE], y_val[:BATCH_SIZE], tok),
                        batch_size=BATCH_SIZE)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            print("🚨 Input IDs (before generation):", batch["input_ids"])
            print("🚨 Labels (before generation):", batch["labels"])

            try:
                stopping_criteria = [StopAfterExtraID1(tok)]
                generated = model.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    max_length=70,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    num_beams=None,
                    no_repeat_ngram_size=3,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria
                )
            except RuntimeError as e:
                print("⚠️ Generation failed due to logits error:", e)
                print(f"Skipping batch with input_ids: {batch['input_ids']}")
                continue
            
            if "labels" not in batch:
                print("⚠️ Skipping batch: 'labels' missing")
                continue

            for o, ref_ids, input_id in zip(generated, batch["labels"], batch["input_ids"]):
                try:
                    if torch.isnan(o).any() or torch.isinf(o).any():
                        print("⚠️ Invalid output (NaN/Inf):", o)
                        continue

                    # Decode forms
                    raw_input = tok.decode(input_id, skip_special_tokens=False)
                    decoded_input = safe_decode_input(tok, input_id) # Approx original
                    cleaned_input = clean_input_sentence(decoded_input)

                    # Output
                    ref_raw = decode_reference(tok, ref_ids)
                    pred = faithful_decode(tok, o).strip()
                    if not pred:
                        print("⚠️ Skipping empty prediction.")
                        continue
                    # Print all views
                    # print(f"📝 Input          : {decoded_input}")
                    print(f"🧼 Cleaned Input  : {cleaned_input}")
                    print(f"🧾 Raw Input      : {raw_input}")
                    print(f"📘 Reference      : {ref_raw}")
                    print(f"🤖 Prediction     : {pred}")
                    print("🧩 Tokens         :", tok.convert_ids_to_tokens(o))
                    print("🆔 Token IDs      :", o.tolist())

                    with open(PRED_LOG_PATH, "a") as f:
                        tag = f"[PREVIEW][Epoch {epoch}]" if epoch is not None else "[PREVIEW]"
                        f.write(f"\n====== {tag} ======\n")
                        f.write(f"{pred}\n")

                    # Decode the tokens before evaluating
                    pred_str = tok.decode(o, skip_special_tokens=True).strip()  # o is the generated tokens
                    ref_str = tok.decode(ref_ids, skip_special_tokens=True).strip()  # ref_ids is the reference tokens

                    preds.append(pred_str)
                    refs.append(ref_str)
                    if detect_hallucination(pred): halluc += 1
                    if not contains_synonym(pred): missing += 1

                except Exception as err:
                    print(f"⚠️ Error during decoding/logging: {err}")
                    continue

            used_synonyms = Counter()
            for pred in preds:
                for token in pred.split():
                    if token in ALL_SYNONYM_TOKENS:
                        used_synonyms[token] += 1

            print("🔍 Synonym Application Stats:")
            for syn, count in used_synonyms.most_common(20):
                print(f"  - {syn}: {count}")
                
# ===================== 🧪 Full Eval =====================

def run_full_evaluation(model, tok, X_val, y_val, output_spans, use_constraints=False, epoch=None):
    print("🧪 Full eval utilizing constraints:", use_constraints, "Total examples:", len(X_val))
    logits_processor = make_logits_processor_per_sample(tok, output_spans) if use_constraints else None
    # Inside run_full_evaluation
    loader = DataLoader(SentenceDataset(X_val[:BATCH_SIZE], y_val[:BATCH_SIZE], tok), batch_size=BATCH_SIZE)
    preds, refs = [], []
    halluc, missing = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            try:
                stopping_criteria = [StopAfterExtraID1(tok)]
                generated = model.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    max_length=70,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    num_beams=None,
                    no_repeat_ngram_size=3,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria
                )
            except RuntimeError as e:
                print("⚠️ Generation failed:", e)
                print(f"Skipping batch with input_ids: {batch['input_ids']}")
                continue

            for o, ref_ids, input_id in zip(generated, batch["labels"], batch["input_ids"]):
                try:
                    if torch.isnan(o).any() or torch.isinf(o).any():
                        print("⚠️ Invalid output (NaN/Inf):", o)
                        continue

                    # Decode forms
                    raw_input = tok.decode(input_id, skip_special_tokens=False)
                    decoded_input = safe_decode_input(tok, input_id) # Approx original
                    cleaned_input = clean_input_sentence(decoded_input)

                    # Output
                    ref_raw = decode_reference(tok, ref_ids).strip()
                    pred = faithful_decode(tok, o)

                    # Print all views
                    # print(f"📝 Input          : {decoded_input}")
                    print(f"🧼 Cleaned Input  : {cleaned_input}")
                    print(f"🧾 Raw Input      : {raw_input}")
                    print(f"📘 Reference      : {ref_raw}")
                    print(f"🤖 Prediction     : {pred}")
                    print("🧩 Tokens         :", tok.convert_ids_to_tokens(o))
                    print("🆔 Token IDs      :", o.tolist())

                    with open(PRED_LOG_PATH, "a") as f:
                        if epoch is not None:
                            f.write(f"\n====== [FULL][Epoch {epoch}] ======\n")
                            f.write(f"{pred}\n")
                        else:
                            f.write(f"[FULL] {pred}\n")

                    preds.append(pred)
                    refs.append(ref_raw)
                    if detect_hallucination(pred): halluc += 1
                    if not contains_synonym(pred): missing += 1

                except Exception as err:
                    print(f"⚠️ Error during decoding/logging: {err}")
                    continue
                
            used_synonyms = Counter()
            for pred in preds:
                for token in pred.split():
                    if token in ALL_SYNONYM_TOKENS:
                        used_synonyms[token] += 1

            print("🔍 Synonym Application Stats:")
            for syn, count in used_synonyms.most_common(20):
                print(f"  - {syn}: {count}")

    # 🛡️ Filter out empty predictions BEFORE metrics
    def is_non_empty(s):
        s = s.strip().lower()
        return s and s not in {"<pad>", "</s>", "<unk>", "<extra_id_0>", "<extra_id_1>"} and "<extra_id_0> <extra_id_1>" not in s

    preds_refs = []
    for i, (p, r) in enumerate(zip(preds, refs)):
        p_clean = p.strip()
        r_clean = r.strip()
        if is_non_empty(p_clean) and is_non_empty(r_clean):
            preds_refs.append((p_clean, r_clean))
        else:
            print(f"⚠️ Skipping trivial or empty prediction/reference at index {i}: pred=“{p_clean}”, ref=“{r_clean}”")

    if not preds_refs:
        print("❌ No valid (non-empty) predictions and references after filtering. Skipping metrics.")
        return {"bleu": 0.0, "rouge": {}, "syn_slot_bleu": {}, "halluc": halluc, "missing_syn": missing}

    preds, refs = zip(*preds_refs)
    preds = list(preds)
    refs = list(refs)
    
    span_matches = 0
    total = 0

    # After preds and refs are finalized
    for pred, gold_span in zip(preds, output_spans):
        if exact_span_match(pred, gold_span):
            span_matches += 1
        total += 1

    span_accuracy = span_matches / total if total else 0.0

    assert all(preds), "⚠️ Empty prediction detected before metric calculation"
    assert all(refs), "⚠️ Empty reference detected before metric calculation"

    # 🔍 Extra debug: log any residual oddities
    for i, (p, r) in enumerate(zip(preds, refs)):
        if not p.strip(): print(f"❗ Empty prediction at {i}")
        if not r.strip(): print(f"❗ Empty reference at {i}")

    # ⚠️ Normalize only for BLEU, not ROUGE
    bleu_refs = normalize_refs(refs, tok)
    
    print(f"🧮 Final eval set size: {len(preds)} predictions, {len(refs)} references")
    assert all(preds) and all(refs), "Empty string detected in final eval input to metrics"
    
    print("📋 Final preds and refs passed to metrics:")
    for i, (p, r) in enumerate(zip(preds, refs)):
        print(f"[{i}] pred: {repr(p)}, ref: {repr(r)}")
        
    for i, pred in enumerate(preds):
        print(f"[CHECK] pred[{i}]: {repr(pred)}")

    return {
        "bleu": compute_bleu(preds, bleu_refs),
        "span_match_accuracy": span_accuracy,
        "rouge": compute_rouge(preds, refs),
        "syn_slot_bleu": compute_synonym_bleu(preds, bleu_refs),
        "halluc": halluc,
        "missing_syn": missing
    }

# ===================== 🏋️ Training =====================

def train_one_model(X_tr, y_tr, X_val, y_val, tok):
    model = get_model(tok)
    Y_span = [SPAN_RE.search(y_).group(1).strip() if SPAN_RE.search(y_) else "" for y_ in y_val]
    # 🔍 Run preview evaluation before training starts (Epoch 0)
    run_preview_evaluation(model, tok, X_val, y_val, use_constraints=False, epoch=0)
    train_dataset = SentenceDataset(X_tr, y_tr, tok)
    best_loss = float("inf")
    best_model_state = None
    loss_curve = []
    syn_ids = [
        tok.convert_tokens_to_ids(w)
        for w in ALL_SYNONYM_TOKENS
        if tok.convert_tokens_to_ids(w) != tok.unk_token_id
    ]

    trainer = SynonymConstrainedTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_dataset,
        synonym_token_ids=syn_ids,
        args=None
    )

    metrics = {"loss_curve": []}

    for epoch in range(1, EPOCHS + 1):
        # 🔁 Run preview eval at each epoch
        run_preview_evaluation(model, tok, X_val, y_val, output_spans=Y_span, use_constraints=False, epoch=0)
        model.train()
        epoch_losses = []
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch_idx, batch in enumerate(loader):
            loss = trainer.compute_loss(model, batch)
            epoch_losses.append(loss.item())
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item()}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()

        avg_loss = np.mean(epoch_losses)
        loss_curve.append(avg_loss)

        # 🧪 Run full eval at each epoch too
        print(f"🧪 Full Evaluation @ Epoch {epoch+1}")
        eval_results = run_full_evaluation(model, tok, X_val, y_val, output_spans=Y_span, use_constraints=False, epoch=epoch)

        for k, v in eval_results.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    metrics["loss_curve"] = loss_curve

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(RESULTS_DIR, "transformer_model_best.pt"))
    return metrics

# ===================== 🚀 Main =====================

def has_repetition(text, proximity=3, window_size=2):
    tokens = text.lower().split()
    counts = Counter(tokens)
    
    # Check if any token appears more than the proximity times within the specified window size
    for i in range(len(tokens) - window_size):
        window = tokens[i:i + window_size]
        if len(set(window)) == 1 and window[0] in counts and counts[window[0]] >= proximity:
            return True
    return False

def main():
    global ALL_SYNONYM_TOKENS
    df, _ = load_datasets()  # Load the dataset

    # Clean the dataset by removing duplicates
    df = df.drop_duplicates(subset=["input_sentence", "output_sentence"])
    
    # 🧹 Drop overly repetitive outputs (Apply the has_repetition function)
    before = len(df)
    df = df[~df["output_sentence"].apply(lambda x: has_repetition(x, proximity=3, window_size=2))]
    after = len(df)
    print(f"🧹 Removed {before - after} samples with repeated tokens (≥3 times within a window of 2)")

    # Now proceed with further dataset cleaning
    df["input_sentence"] = df["input_sentence"].astype(str).apply(clean_input_sentence)
    df["output_sentence"] = df["output_sentence"].astype(str).apply(reformat_output_sentence)
    
    # Continue with tokenization, model setup, training, etc.
    X, y = df["input_sentence"].tolist(), df["output_sentence"].tolist()
    tok = get_tok()
    assert os.path.exists("./t5_base_model/config.json"), "Model config path is wrong!"
    tok.add_tokens(list(ALL_SYNONYM_TOKENS))
    ALL_SYNONYM_TOKENS = filter_valid_synonyms(tok)
    print("🧠 Final Synonym Tokens Used:")
    for s in sorted(ALL_SYNONYM_TOKENS):
        print(f"  - {s} → {tok.tokenize(s)}")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    all_fold = {}

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        print(f"🚀 Round {fold}/{NUM_FOLDS}")

        # Only run preview evaluation for the first round
        if fold == 1:
            # Do NOT run preview here; run inside train_one_model after epoch 1
            pass
            preview_model = get_model(tok)
            # Run preview evaluation with no constraints for the first fold
            run_preview_evaluation(preview_model, tok, [X[i] for i in va], [y[i] for i in va], use_constraints=False)

        metrics = train_one_model([X[i] for i in tr], [y[i] for i in tr],
                                [X[i] for i in va], [y[i] for i in va], tok)
        all_fold[f"fold_{fold}"] = metrics

        plt.figure()
        plt.plot(metrics["loss_curve"])
        plt.title(f"Loss Curve - Round {fold}")
        plt.savefig(f"{RESULTS_DIR}/visualizations/plots/transformer/loss_curve_round{fold}.png")
        plt.close()

    with open(f"{RESULTS_DIR}/json/transformer_metrics.json", "w") as f:
        json.dump(all_fold, f, indent=2)
        
        # Compute and save summary metrics (mean/std)
        metric_data = []
        for fold, metrics in all_fold.items():
            row = {k: v[-1] if k == "loss_curve" and isinstance(v, list) else v for k, v in metrics.items()}
            row["fold"] = fold
            metric_data.append(row)

        df = pd.DataFrame(metric_data)
        df.to_csv(f"{RESULTS_DIR}/csv/per_round_scores.csv", index=False)

        numeric_df = df.drop(columns=["fold"]).select_dtypes(include=[np.number])
        summary = numeric_df.agg(['mean', 'std'])
        summary.to_csv(f"{RESULTS_DIR}/csv/summary_metrics.csv")

if __name__ == "__main__":
    main()
