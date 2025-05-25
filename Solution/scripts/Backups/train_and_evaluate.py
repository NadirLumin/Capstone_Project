# ===================== 🛑 Block TensorFlow Auto-Import =====================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== 🤖 Hugging Face Transformers ======================
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    LogitsProcessor, LogitsProcessorList, TopKLogitsWarper,
    Trainer, TrainingArguments, PrinterCallback
)

# ========================= 🔧  Standard Library  ============================
import json, re, sys, time, logging, inspect, string, gc
from collections import Counter

# ======================= 📦  3rd‑Party Libraries  ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

# ========================= 🔨  Local Modules  ===============================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from data_loader import load_datasets
from metrics import compute_bleu, compute_rouge, compute_synonym_bleu

# ========================== ⚙️  CONFIGURATION  ==============================
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDATED_PATH    = "./data/cleaned_synonyms_data.csv"
RESULTS_DIR      = "./results"
EPOCHS, BATCH_SZ = 10, 16
FOLDS, ACC_STEPS = 5, 3
LR               = 2e-5

os.makedirs(f"{RESULTS_DIR}/plots/transformer",  exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/csv",                exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/json",               exist_ok=True)

# =================== 🧠 Hard‑coded synonym map  =============================
def _norm(t):
    t = re.sub(r"[“”‘’]", lambda m: '"' if m.group(0) in "“”" else "'", t)
    return re.sub(r"\(s\)", "", t).strip().strip('"').lower()

df_syn = pd.read_csv(OUTDATED_PATH)
df_syn["Outdated Term"]      = df_syn["Outdated Term"].astype(str).apply(_norm)
df_syn["Exuberant Synonyms"] = df_syn["Exuberant Synonyms"].astype(str).apply(_norm)

hardcoded_map = {row["Outdated Term"]: row["Exuberant Synonyms"].split(",")[0].strip()
                 for _, row in df_syn.iterrows()
                 if row["Outdated Term"] and row["Exuberant Synonyms"]}

OUTDATED_TERMS       = list(hardcoded_map.keys())
ALL_SYNONYM_TOKENS   = set(hardcoded_map.values())
lemmatizer           = WordNetLemmatizer()

print("🔍 Allowed synonym tokens:", sorted(ALL_SYNONYM_TOKENS))

# ===================== 🔠 Tokeniser / model helpers =========================
def get_tok(path="./t5_base_model"):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.add_special_tokens(
        {'unk_token': '<unk>', 'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>']}
    )
    return tok

def get_model(tok, path="./t5_base_model"):
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
    model.resize_token_embeddings(len(tok))
    return model.to(DEVICE)

def filter_valid_synonyms(tok):
    return {s for s in ALL_SYNONYM_TOKENS if all(tok.convert_tokens_to_ids(t) != tok.unk_token_id
                                                 for t in tok.tokenize(s))}

# ===================== 🔧  Text clean‑ups / helpers =========================
SPAN_RE = re.compile(r"<extra_id_0>\s*(.*?)\s*<extra_id_1>", re.IGNORECASE)

def apply_hardcoded_synonym(txt):
    def _sub(m):
        term = m.group(1).lower().strip()
        return f" <extra_id_0> {hardcoded_map.get(term, term)} <extra_id_1> "
    return SPAN_RE.sub(_sub, txt).strip()

def faithful_decode(tok, ids):
    tokens, last = [], None
    for t in tok.convert_ids_to_tokens(ids, skip_special_tokens=True):
        if t != last:
            tokens.append(t)
            last = t
    out = tok.convert_tokens_to_string(tokens).replace("<unk>", "").strip()
    return apply_hardcoded_synonym(out)

def clean_input_sentence(s):
    s = str(s)
    s = re.sub(r"(\s*<extra_id_0>\s*)+", " <extra_id_0> ", s)
    s = re.sub(r"(\s*<extra_id_1>\s*)+", " <extra_id_1> ", s)
    return re.sub(r"\s+", " ", s).strip()
# ===================== 🔒  Synonym‑constraint utils  ========================
class SynonymOnlyBetweenTagsProcessor(LogitsProcessor):
    def __init__(self, syn_ids, tok, top_k=12):
        self.syn_ids = set(syn_ids)
        self.extra_ids = {tok.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)}
        self.topk = TopKLogitsWarper(top_k)
        self.inside, self.just_extra = False, False

    def __call__(self, input_ids, scores):
        last = input_ids[0, -1].item()
        if last in self.extra_ids:
            self.inside, self.just_extra = not self.inside, True
            return scores
        if self.just_extra: self.just_extra = False

        if self.inside:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, list(self.syn_ids)] = 0
            scores += mask
            return self.topk(input_ids, scores)
        return scores

def make_logits_processor(tok):
    syn_ids = [tok.convert_tokens_to_ids(s) for s in ALL_SYNONYM_TOKENS if tok.convert_tokens_to_ids(s) != tok.unk_token_id]
    return LogitsProcessorList([SynonymOnlyBetweenTagsProcessor(syn_ids, tok)])

# ===================== 🔠  Dataset wrapper  ================================
class SentenceDataset(Dataset):
    def __init__(self, X, y, tok, max_len=70):
        self.X, self.y, self.tok, self.max_len = X, y, tok, max_len
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        enc = self.tok(clean_input_sentence(self.X[idx]),
                       max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        dec = self.tok(self.y[idx], max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": dec["input_ids"].squeeze(0)}

# ---------------------  preview & full‑eval share config  ------------------
GEN_KWARGS = dict(
    max_length   = 90,
    do_sample    = True,
    top_k        = 30,
    top_p        = 0.75,
    temperature  = 0.5,
)

# ===================== 🧪 Preview evaluation  ==============================
def run_preview_eval(model, tok, X_val, y_val):
    dl = DataLoader(SentenceDataset(X_val[:BATCH_SZ], y_val[:BATCH_SZ], tok),
                    batch_size=BATCH_SZ)
    proc = make_logits_processor(tok)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            outs = model.generate(**{k: v.to(DEVICE) for k, v in batch.items() if k!="labels"},
                                  logits_processor=proc, **GEN_KWARGS)
            for j, o in enumerate(outs):
                raw_input    = X_val[i * BATCH_SZ + j]
                cleaned_input = clean_input_sentence(raw_input)
                ref_text     = y_val[i * BATCH_SZ + j]
                pred_text    = faithful_decode(tok, o)
                print(f"\n🔢 {i * BATCH_SZ + j}")
                print(f"📝 Raw Input     : {raw_input}")
                print(f"🧼 Cleaned Input : {cleaned_input}")
                print(f"📘 Reference     : {ref_text}")
                print(f"🤖 Prediction    : {pred_text}")

# ===================== 🧪 Full evaluation  ================================
def run_full_eval(model, tok, X_val, y_val):
    dl = DataLoader(SentenceDataset(X_val, y_val, tok), batch_size=BATCH_SZ)
    proc = make_logits_processor(tok)
    preds, refs = [], []
    halluc, missing = 0, 0

    model.eval()
    with torch.no_grad():
        for batch in dl:
            outs = model.generate(**{k: v.to(DEVICE) for k, v in batch.items() if k!="labels"},
                                  logits_processor=proc, **GEN_KWARGS)
            for pred_ids, ref_ids in zip(outs, batch["labels"]):
                pred = faithful_decode(tok, pred_ids)
                ref  = faithful_decode(tok, ref_ids)
                preds.append(pred)
                refs.append(ref)
                if any(term in pred.lower() for term in OUTDATED_TERMS): halluc += 1
                if not any(s in pred.lower() for s in ALL_SYNONYM_TOKENS): missing += 1

    return {
        "bleu": compute_bleu(preds, refs),
        "rouge": compute_rouge(preds, refs),
        "syn_slot_bleu": compute_synonym_bleu(preds, refs),
        "halluc": halluc,
        "missing_syn": missing
    }

# ===================== 🏋️  Train‑one‑round  ================================
def train_one_round(Xtr, ytr, Xva, yva, tok, fold):
    model  = get_model(tok)
    ds_tr  = SentenceDataset(Xtr, ytr, tok)
    ds_va  = SentenceDataset(Xva, yva, tok)

    class SynonymConstrainedTrainer(Trainer):
        def __init__(self, *args, synonym_token_ids=None, tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.synonym_token_ids = synonym_token_ids
            self._tokenizer = tokenizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            if "decoder_input_ids" not in inputs:
                inputs["decoder_input_ids"] = model._shift_right(labels)
            outputs = model(**inputs)
            logits = outputs.logits.clone()

            for i in range(logits.size(0)):
                lab = labels[i]
                try:
                    st = (lab == self._tokenizer.convert_tokens_to_ids("<extra_id_0>")).nonzero()[0].item()+1
                    ed = (lab == self._tokenizer.convert_tokens_to_ids("<extra_id_1>")).nonzero()[0].item()
                    logits[i, st:ed, :] = float("-inf")
                    logits[i, st:ed, self.synonym_token_ids] = 0
                except: pass

            loss = nn.CrossEntropyLoss(ignore_index=self._tokenizer.pad_token_id)(
                logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # ✅ Apply correctly
    syn_ids = [tok.convert_tokens_to_ids(s) for s in ALL_SYNONYM_TOKENS]

    trainer = SynonymConstrainedTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results/checkpoints",
        per_device_train_batch_size=BATCH_SZ,
        per_device_eval_batch_size=BATCH_SZ,
        gradient_accumulation_steps=ACC_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        eval_strategy="epoch",  # 🔁 updated here
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=5,
        report_to=[]
    ),  # <-- Close the TrainingArguments parentheses here
    train_dataset=ds_tr,
    eval_dataset=ds_va,
    tokenizer=tok,
    callbacks=[PrinterCallback()],
    synonym_token_ids=syn_ids
    )  # <-- Close the SynonymConstrainedTrainer parentheses here

    # 🏋️ Train and log
    trainer.train()
    
    loss_history = trainer.state.log_history
    loss_curve = [e["loss"] for e in loss_history if "loss" in e]

    plt.figure()
    plt.plot(loss_curve)
    plt.title(f"Loss Curve - Round {fold}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(f"{RESULTS_DIR}/plots/transformer/loss_curve_round{fold}.png")
    plt.close()

    eval_metrics = trainer.evaluate()
    eval_metrics.update(epoch=EPOCHS, fold=fold,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    pd.DataFrame([eval_metrics]).to_csv(
        f"{RESULTS_DIR}/csv/eval_metrics_round{fold}.csv", index=False)

    return model

# ===================== 🚀 Main Entrypoint ================================
def main():
    global ALL_SYNONYM_TOKENS
    df,_  = load_datasets()
    df    = df.drop_duplicates(subset=["input_sentence","output_sentence"])
    df["input_sentence"] = df["input_sentence"].astype(str).apply(clean_input_sentence)

    tok = get_tok(); tok.add_tokens(list(ALL_SYNONYM_TOKENS))
    ALL_SYNONYM_TOKENS = filter_valid_synonyms(tok)

    print("\n🔎 Synonym tokenisation check:")
    for s in sorted(ALL_SYNONYM_TOKENS): print(f"{s} → {tok.tokenize(s)}")

    # Sanity check
    print("\n🔬 One-off sanity check:")
    m = get_model(tok)
    test_ids = m.generate(**tok("How to <extra_id_0> manage <extra_id_1> risks?", return_tensors="pt").to(DEVICE),
                          logits_processor=make_logits_processor(tok), **GEN_KWARGS)[0]
    print("🧠 Output:", faithful_decode(tok, test_ids))

    X, y = df["input_sentence"].tolist(), df["output_sentence"].tolist()
    kf   = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    all_metrics = {}

    for fold,(tr,va) in enumerate(kf.split(X),1):
        print(f"\n🚀 Round {fold}/{FOLDS}")
        if fold == 1:
            print("🧪 Running preview eval...")
            run_preview_eval(m, tok, [X[i] for i in va], [y[i] for i in va])
        model = train_one_round([X[i] for i in tr],[y[i] for i in tr],
                               [X[i] for i in va],[y[i] for i in va],
                               tok, fold)
        metrics = run_full_eval(model, tok, [X[i] for i in va], [y[i] for i in va])
        all_metrics[f"fold_{fold}"] = metrics

        plt.figure()
        plt.title(f"Loss Curve - Round {fold}")
        plt.savefig(f"{RESULTS_DIR}/plots/transformer/loss_curve_round{fold}.png")
        plt.close()
        
        # 📊 Save summary statistics
        df_metrics = pd.DataFrame.from_dict(all_metrics, orient="index")
        df_metrics.to_csv(f"{RESULTS_DIR}/csv/aggregate_metrics.csv")

        summary = {
            "mean": df_metrics.mean(numeric_only=True).to_dict(),
            "std":  df_metrics.std(numeric_only=True).to_dict()
        }

        with open(f"{RESULTS_DIR}/json/aggregate_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n📈 Aggregate Metrics:")
        for k in summary["mean"]:
            print(f"{k}: mean={summary['mean'][k]:.4f}, std={summary['std'][k]:.4f}")

    with open(f"{RESULTS_DIR}/json/transformer_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()
