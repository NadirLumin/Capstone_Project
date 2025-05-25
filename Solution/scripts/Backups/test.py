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
    PrinterCallback,
    get_cosine_schedule_with_warmup,
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
    f.write("")  # Clear previous predictions
EPOCHS, BATCH_SIZE = 10, 16
NUM_FOLDS, ACCUM_STEPS = 5, 3
LEARNING_RATE = 2e-5
frequency_proximity = 5
os.makedirs(f"{RESULTS_DIR}/json", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/plots/transformer", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/csv", exist_ok=True)
lemmatizer = WordNetLemmatizer()

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
        if all(tok.convert_tokens_to_ids(t) != tok.unk_token_id for t in tokens):
            valid.add(s)
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
    return ref

def faithful_decode(tok, ids):
    tokens = tok.convert_ids_to_tokens(ids, skip_special_tokens=True)
    decoded = tok.convert_tokens_to_string(tokens)
    decoded = decoded.replace("<unk>", "").replace("<s>", "").replace("</s>", "").strip()
    return apply_hardcoded_synonym(decoded)

# def clean_input_sentence(s):
#     if isinstance(s, (torch.Tensor, np.ndarray)):
#         s = ''.join([chr(i) for i in s if i != 0])
#     elif not isinstance(s, str):
#         return s

#     # Remove duplicate <extra_id_*> tags and standardize spacing
#     s = re.sub(r"(\s*<extra_id_0>\s*)+", " <extra_id_0> ", s)
#     s = re.sub(r"(\s*<extra_id_1>\s*)+", " <extra_id_1> ", s)

#     s = re.sub(r"<unk>", "", s, flags=re.IGNORECASE)
#     s = re.sub(r"\s*(<extra_id_0>|<extra_id_1>)\s*", r" \1 ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

def reformat_output_sentence(y: str) -> str:
    # Ensure space *after* <extra_id_0> and *before* <extra_id_1>
    y = re.sub(r"<extra_id_0>\s*", "<extra_id_0> ", y)
    y = re.sub(r"\s*<extra_id_1>", " <extra_id_1>", y)

    # Clean up extra spacing and unknown tokens
    y = re.sub(r"<unk>", "", y, flags=re.IGNORECASE)
    y = re.sub(r"\s+", " ", y).strip()

    return y

def normalize_spacing_around_extra_ids(txt: str) -> str:
    txt = re.sub(r"\s*(<extra_id_\d+>)\s*", r" \1 ", txt)
    return re.sub(r"\s+", " ", txt).strip()

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

# ===================== 🔠 Dataset =====================

class SentenceDataset(Dataset):
    def __init__(self, X, y, tok, max_len=70):
        self.X, self.y, self.tok, self.max_len = X, y, tok, max_len

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        # x = clean_input_sentence(self.X[idx])
        x = self.X[idx]
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

        for i in range(logits.size(0)): # Batch dimension
            label_ids = labels[i]
            start, end = None, None
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

        loss_fct = nn.CrossEntropyLoss(ignore_index=self._tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class SynonymOnlyBetweenTagsProcessor(LogitsProcessor):
    def __init__(self, synonym_token_ids, tokenizer, top_k=10):
        self.syn_ids = set(synonym_token_ids)
        self.extra_ids = {
            tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            for i in range(100)
        }
        self.topk_warper = TopKLogitsWarper(top_k)
        self.inside = False
        self.last_was_extra = False

    def __call__(self, input_ids, scores):
        last = input_ids[0, -1].item()

        if last in self.extra_ids:
            self.inside = not self.inside
            self.last_was_extra = True
            return scores
        elif self.last_was_extra:
            self.last_was_extra = False
        
        if self.inside:
            allowed_tokens = list(self.syn_ids)
            if not allowed_tokens:
                print("⚠️ No allowed synonym tokens — skipping masking.")
                return scores # Avoid full masking if no synonyms

            # Mask all first
            mask = torch.full_like(scores, float('-inf'))
            mask[:, allowed_tokens] = 0.0
            scores = scores + mask

            # Replace all -inf with uniform log-probs if totally masked
            if torch.isinf(scores).all():
                print("🛑 All scores were -inf — recovering with uniform mask.")
                scores = torch.full_like(scores, float("-inf"))
                scores[:, allowed_tokens] = 0.0

            scores = scores.masked_fill(torch.isnan(scores), float('-inf'))

            # Fallback if scores are all invalid
            if torch.isinf(scores).all():
                print("🛑 All scores are -inf, forcing uniform logits across allowed synonyms")
                scores = torch.full_like(scores, float("-inf"))
                scores[:, allowed_tokens] = 0.0 # Uniform log-prob

            scores = scores.masked_fill(torch.isnan(scores), float("-inf"))

            # Ensure diversity in the predictions and add fallback for invalid scenarios
            if torch.isnan(scores).any() or torch.isinf(scores).all():
                print("🔥 Final fallback triggered — assigning uniform probability across allowed synonyms.")
                scores = torch.full_like(scores, 0.0)
                if allowed_tokens:
                    scores[:, :] = float("-inf")
                    scores[:, allowed_tokens] = 0.0
                else:
                    print("🚨 No allowed synonym tokens — fallback returns uniform over full vocab (unsafe)")

            return self.topk_warper(input_ids, scores)

        return scores
    
def make_logits_processor(tok):
    syn_ids = [
        tok.convert_tokens_to_ids(w)
        for w in ALL_SYNONYM_TOKENS
        if tok.convert_tokens_to_ids(w) != tok.unk_token_id
    ]
    return LogitsProcessorList([
        SynonymOnlyBetweenTagsProcessor(syn_ids, tok, top_k=12)
    ])

# ===================== 🧪 Preview Eval =====================

def run_preview_evaluation(model, tok, X_val, y_val):
    loader = DataLoader(SentenceDataset(X_val[:BATCH_SIZE], y_val[:BATCH_SIZE], tok),
                        batch_size=BATCH_SIZE)
    proc = make_logits_processor(tok)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            try:
                # Generate predictions deterministically
                outs = model.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    max_length=90,
                    do_sample=True, # Enable sampling
                    top_k=30, # Consider the top k probabilities
                    top_p=0.75, # Utilize nucleus sampling for more diversity
                    temperature=0.5, # Controls randomness
                    logits_processor=proc,
                )
            except RuntimeError as e:
                # Log the error, and log the problematic inputs for further inspection
                print("⚠️ Generation failed due to invalid probabilities:", e)
                print(f"Skipping batch with input_ids: {batch['input_ids']}")
                continue # Skip this batch and move on to the next one

            # Now, decode the outputs and log the inputs, references, and predictions
            for o, ref_ids, input_id in zip(outs, batch["labels"], batch["input_ids"]):
                try:
                    # Check for NaN or Inf in the generated output before proceeding
                    if torch.isnan(o).any() or torch.isinf(o).any():
                        print("⚠️ Invalid output detected (NaN/Inf):", o)
                        print(f"Skipping this output for input: {input_id}")
                        continue # Skip this output and move on to the next one

                    # Decode the raw input and cleaned input
                    inp = safe_decode_input(tok, input_id)
                    print(f"🧾 Decoded Raw Input : {tok.decode(input_id, skip_special_tokens=False)}")
                    # print(f"🧼 Cleaned Input : {clean_input_sentence(inp)}")
                    print(f"🧼 Input        : {inp}")

                    # Decode reference sentences
                    ref_raw = decode_reference(tok, ref_ids)
                    print(f"📘 Reference   : {ref_raw}")
                    # ref_clean = decode_ref_clean(tok, ref_ids)
                    # print(f"📙 Ref (clean): {ref_clean}")
                    
                    # Decode the model prediction
                    pred = faithful_decode(tok, o)
                    with open(PRED_LOG_PATH, "a") as f:
                        f.write(pred + "\n")
                    print(f"📝 Raw input IDs : {input_id.tolist()}")
                    print(f"📝 Input       : {inp}")
                    print(f"📘 Reference   : {ref_raw}")
                    print(f"🤖 Prediction  : {pred}")
                    
                    # Log the tokens and token IDs
                    print("🧩 Tokens      :", tok.convert_ids_to_tokens(o))
                    print("🆔 Token IDs   :", o.tolist())
                
                except Exception as err:
                    # Log any errors that happen during decoding or logging
                    print(f"⚠️ Error during decoding or logging: {err}")
                    continue # Skip this batch if an error occurs
                
# ===================== 🧪 Full Eval =====================

def run_full_evaluation(model, tok, X_val, y_val):
    loader = DataLoader(SentenceDataset(X_val, y_val, tok), batch_size=BATCH_SIZE)
    proc = make_logits_processor(tok)
    preds, refs = [], []
    halluc, missing = 0, 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            try:
                # Generate predictions deterministically
                outs = model.generate(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    max_length=90,
                    do_sample=True, # Enable sampling
                    top_k=30, # Consider the top k probabilities
                    top_p=0.75, # Utilize nucleus sampling for more diversity
                    temperature=0.5, # Controls randomness
                    logits_processor=proc,
                )
            except RuntimeError as e:
                print("⚠️ Generation failed due to invalid probabilities:", e)
                print(f"Skipping batch with input_ids: {batch['input_ids']}")
                continue

            for pred_ids, ref_ids, input_id in zip(outs, batch["labels"], batch["input_ids"]):
                pred = faithful_decode(tok, pred_ids)
                ref_raw = decode_reference(tok, ref_ids)
                ref_raw = normalize_spacing_around_extra_ids(ref_raw)
                ref_raw = strip_special_tokens(ref_raw)
                ref_raw = decode_reference(tok, ref_ids)
                print(f"📘 Reference   : {ref_raw}")
                inp = safe_decode_input(tok, input_id)
                raw_input = tok.decode(input_id, skip_special_tokens=False)

                print(f"🧾 Raw Input    : {raw_input}")
                # print(f"🧼 Cleaned Input: {inp}")
                print(f"🧼 Input        : {inp}")
                print(f"🤖 Prediction   : {pred}")

                with open(PRED_LOG_PATH, "a") as f:
                    f.write(f"[FULL EVAL] {pred}\n")

                preds.append(pred)
                refs.append(ref_raw)  # Or ref_clean if you’re scoring against clean refs
                if detect_hallucination(pred): halluc += 1
                if not contains_synonym(pred): missing += 1

    return {
        "bleu": compute_bleu(preds, refs),
        "rouge": compute_rouge(preds, refs),
        "syn_slot_bleu": compute_synonym_bleu(preds, refs),
        "halluc": halluc,
        "missing_syn": missing
    }

# ===================== 🏋️ Training =====================

def train_one_model(X_tr, y_tr, X_val, y_val, tok):
    model = get_model(tok)
    train_dataset = SentenceDataset(X_tr, y_tr, tok)
    val_dataset = SentenceDataset(X_val, y_val, tok)

    # Create synonym token ID list
    synonym_token_ids = [tok.convert_tokens_to_ids(w) for w in ALL_SYNONYM_TOKENS if tok.convert_tokens_to_ids(w) != tok.unk_token_id]

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results/checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=5,
        report_to=[], # Disable wandb/hf logging
        logging_dir="./results/logs",
    )
    
    # Trainer with constrained loss
    trainer = SynonymConstrainedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tok,
        synonym_token_ids=synonym_token_ids,
        callbacks=[PrinterCallback()],
    )

    # 🚀 Train
    trainer.train()
    trainer.evaluate()
    eval_metrics = trainer.evaluate()
    
    # Save eval metrics for this round
    eval_metrics["epoch"] = EPOCHS  # Add epoch info for clarity
    eval_metrics_path = os.path.join(RESULTS_DIR, "csv", "eval_metrics_per_round.csv")
    
    # Append or create CSV
    if not os.path.exists(eval_metrics_path):
        pd.DataFrame([eval_metrics]).to_csv(eval_metrics_path, index=False)
    else:
        existing = pd.read_csv(eval_metrics_path)
        updated = pd.concat([existing, pd.DataFrame([eval_metrics])], ignore_index=True)
        updated.to_csv(eval_metrics_path, index=False)

    # 🔁 Log post-training sample predictions
    print("📊 Logging training sample predictions...")
    model.eval()
    sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(sample_loader):
        if i >= 4: break  # Preview 4 examples
        raw_input = tok.decode(batch["input_ids"][0], skip_special_tokens=False)
        # cleaned_input = safe_decode_input(tok, batch["input_ids"][0])
        ref_raw = decode_reference(tok, batch["labels"][0])
        pred = faithful_decode(tok, model.generate(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            max_length=90,
            do_sample=True, # Enable sampling
            top_k=30, # Consider the top k probabilities
            top_p=0.75, # Utilize nucleus sampling for more diversity
            temperature=0.5, # Controls randomness
            logits_processor=proc,
        )
        )

        print(f"🧾 Raw Input    : {raw_input}")
        # print(f"🧼 Cleaned Input: {cleaned_input}")
        print(f"🧼 Input        : {inp}")
        print(f"📘 Reference    : {ref_raw}")
        print(f"🤖 Prediction   : {pred}")

        with open(PRED_LOG_PATH, "a") as f:
            f.write(pred + "\n")

    # 🧪 Full eval
    print("🧪 Running full evaluation...")
    metrics = run_full_evaluation(model, tok, X_val, y_val)
    metrics["loss_curve"] = [] # You can populate this via manual tracking if desired
    return metrics

# ===================== 🚀 Main =====================

def main():
    global ALL_SYNONYM_TOKENS
    df, _ = load_datasets()
    df = df.drop_duplicates(subset=["input_sentence", "output_sentence"])
    # df["input_sentence"] = df["input_sentence"].astype(str).apply(clean_input_sentence)
    df["output_sentence"] = df["output_sentence"].astype(str).apply(reformat_output_sentence)
    X, y = df["input_sentence"].tolist(), df["output_sentence"].tolist()
    tok = get_tok()
    assert os.path.exists("./t5_base_model/config.json"), "Model config path is wrong!"
    tok.add_tokens(list(ALL_SYNONYM_TOKENS))
    ALL_SYNONYM_TOKENS = filter_valid_synonyms(tok)
    print(f"✅ Valid synonym tokens retained: {len(ALL_SYNONYM_TOKENS)}")
    print(f"Examples: {list(ALL_SYNONYM_TOKENS)[:10]}")
    for s in sorted(ALL_SYNONYM_TOKENS):
        tid = tok.convert_tokens_to_ids(s)
        if tid == tok.unk_token_id:
            print(f"⚠️ Synonym '{s}' is tokenized as <unk>")
            
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    all_fold = {}

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        print(f"🚀 Round {fold}/{NUM_FOLDS}")

        # Only run preview evaluation for the first round
        if fold == 1:
            print("🧪 Running preview evaluation before training...")
            preview_model = get_model(tok)
            run_preview_evaluation(preview_model, tok, [X[i] for i in va], [y[i] for i in va])

        metrics = train_one_model([X[i] for i in tr], [y[i] for i in tr],
                                [X[i] for i in va], [y[i] for i in va], tok)
        all_fold[f"fold_{fold}"] = metrics

        plt.figure()
        plt.plot(metrics["loss_curve"])
        plt.title(f"Loss Curve - Round {fold}")
        plt.savefig(f"{RESULTS_DIR}/plots/transformer/loss_curve_round{fold}.png")
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
