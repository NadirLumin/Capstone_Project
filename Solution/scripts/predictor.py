import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import logging
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import inflect
p = inflect.engine()

# === Configuration ===
MODEL_PATH = "./results/model_round_5"
OUTDATED_TERMS_PATH = "./data/cleaned_synonyms_data.csv"
MAX_TOKENS = 250  # Prototyping token limit
MAX_CHARS = 250   # Prototyping character limit

# === Logging Setup ===
logging.basicConfig(
    filename="prediction_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
lemmatizer = WordNetLemmatizer()

LABEL2ID = {"O": 0, "OUTDATED": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# === Load Synonym Mapping ===
def _norm(term):
    term = re.sub(r"[“”‘’]", '"', term)
    return re.sub(r"\(s\)", "", term).strip().strip('"').lower()

df_syn = pd.read_csv(OUTDATED_TERMS_PATH)
df_syn["Outdated Term"] = df_syn["Outdated Term"].astype(str).apply(_norm)
df_syn["POS"] = df_syn["POS"].astype(str).str.lower().str.strip()
df_syn["Exuberant Synonyms"] = df_syn["Exuberant Synonyms"].astype(str).apply(_norm)

hardcoded_synonym_map = {
    (row["Outdated Term"], row["POS"]): row["Exuberant Synonyms"].split(",")[0].strip()
    for _, row in df_syn.iterrows()
    if row["Outdated Term"] and row["Exuberant Synonyms"] and row["POS"]
}
OD_TERMS_POS = set(hardcoded_synonym_map.keys())
OD_TERMS = set([k[0] for k in OD_TERMS_POS])

print("\n[DEBUG] Keys in hardcoded_synonym_map (first 20):")
for k in list(hardcoded_synonym_map.keys())[:20]:
    print("   ", k)

print("\n[DEBUG] Checking if ('user', 'noun') is in map:")
key = ('user', 'noun')
print("   Exists?", key in hardcoded_synonym_map)
if key in hardcoded_synonym_map:
    print("   Synonym:", hardcoded_synonym_map[key])
else:
    print("   Not found in mapping.")

# === Helpers ===
def penn_to_simple(pos):
    if pos == 'MD':
        return 'modal'
    if pos.startswith('N'):
        return 'noun'
    if pos.startswith('V'):
        return 'verb'
    if pos.startswith('J'):
        return 'adj'
    if pos.startswith('R'):
        return 'adv'
    return pos.lower()

def simple_lemmatize(word):
    word = word.lower().strip()
    if word.endswith("'s"):
        word = word[:-2]
    elif word.endswith("s'"):
        word = word[:-2]
    elif word.endswith("es") and word[:-2] in OD_TERMS:
        word = word[:-2]
    elif word.endswith("s") and word[:-1] in OD_TERMS:
        word = word[:-1]
    elif word.endswith("ed") and word[:-2] in OD_TERMS:
        word = word[:-2]
    elif word.endswith("ing") and word[:-3] in OD_TERMS:
        word = word[:-3]
    return word

def custom_tokenize(text):
    return re.findall(r"<extra_id_\d+>|[\w']+|[.,!?;:]", text)

def smart_join(tokens):
    result = ""
    no_space_before = {'.', ',', '!', '?', ';', ':'}
    for i, token in enumerate(tokens):
        prev_token = tokens[i - 1] if i > 0 else None
        next_token = tokens[i + 1] if i < len(tokens) - 1 else None

        # Always add a tag with a trailing space if not followed by another tag or punctuation
        if token.startswith("<extra_id_"):
            if i > 0 and not result.endswith(" "):
                result += " "
            result += token
            # Add space after tag if next token is a word (not punctuation or tag)
            if next_token and not (next_token in no_space_before or next_token.startswith("<extra_id_")):
                result += " "
            elif next_token is None:
                result += " "
        elif prev_token and prev_token.startswith("<extra_id_"):
            # Previous was a tag, space already handled above
            result += token
        elif token in no_space_before:
            # No space before punctuation
            result += token
        else:
            # Regular case: add space before token
            if not result.endswith(" "):
                result += " "
            result += token

    # Collapse extra spaces
    result = re.sub(r"\s+", " ", result)
    # Remove space before punctuation (like " word ." => " word.")
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return result.strip()

def insert_extra_id_tags(sentence):
    tokens = custom_tokenize(sentence)
    print(tokens)
    outdated_bases = set([k[0] for k in hardcoded_synonym_map.keys()])
    new_tokens = []
    tag_idx = 0
    for word in tokens:
        base = simple_lemmatize(word)
        if base in outdated_bases:
            new_tokens.append(f"<extra_id_{tag_idx}>")
            new_tokens.append(word)
            new_tokens.append(f"<extra_id_{tag_idx+1}>")
            tag_idx += 2  # Always increment by two (open and close tag)
        else:
            new_tokens.append(word)
    joined = " ".join(new_tokens)
    joined = re.sub(r"\s+([.,!?;:])", r"\1", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined

def match_capitalization(src, tgt):
    if src.isupper():
        return tgt.upper()
    elif src.istitle():
        return tgt.title()
    else:
        return tgt.lower()

def strip_extra_id_tags(text):
    # Remove all <extra_id_x> tags, and normalize whitespace
    text = re.sub(r"\s*<extra_id_\d+>\s*", " ", text)
    text = re.sub(r"\s+", " ", text) # Normalize extra spaces
    # Remove any space before punctuation
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text.strip()

# === Adversarial Detection Function ===

# def detect_adversarial(text):
#     # Example rules; expand as needed!
#     if text.count("<extra_id_") > 10:  # excessive special tags
#         return True
#     if "|||" in text:
#         return True
#     if len(custom_tokenize(text)) > MAX_TOKENS:
#         return True
#     return False

# === Logging Functions ===

def log_event(event_type, text):
    logging.info(f"{event_type}: {text}")

def log_prediction(input_text, output_text):
    logging.info(f"Prediction - Input: {input_text} | Output: {output_text}")

# === Swap Outdated Logic ===

def swap_outdated(sentence, model, provided_pos_map=None, return_indices=False, adversarial_spans=None):
    print(f"\n[DEBUG] Input sentence: '{sentence}'")
    tokens = custom_tokenize(sentence)
    print(f"[DEBUG] Custom tokens: {tokens}")
    pos_tags = nltk.pos_tag(tokens)
    print(f"[DEBUG] POS tags: {pos_tags}")
    words = tokens

    for idx, word in enumerate(words):
        base_term = simple_lemmatize(word)
        pos = pos_tags[idx][1]
        pos_simple = penn_to_simple(pos)
        print(f"[DEBUG] At idx {idx}: word '{word}' | base_term: '{base_term}', pos: '{pos}', pos_simple: '{pos_simple}'")

    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = enc.word_ids()
    print(f"[DEBUG] Tokenizer word_ids: {word_ids}")
    logits = model(**enc).logits.squeeze(0)
    preds = logits.argmax(dim=-1).cpu().tolist()

    print(f"[DEBUG] Inference tokens: {words}")
    print(f"[DEBUG] word_ids: {word_ids}")
    print(f"[DEBUG] Model preds: {preds} ({[ID2LABEL.get(x, '?') for x in preds]})")
    for idx, (wid, word) in enumerate(zip(word_ids, words)):
        pred_val = ID2LABEL.get(preds[idx], '?') if wid is not None and idx < len(preds) else '?'
        print(f"  idx={idx} word='{word}' word_id={wid} pred={pred_val}")
    print(f"[DEBUG] Model preds: {preds} ({[ID2LABEL.get(x, '?') for x in preds]})")
    target_word_indices = set()
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if preds[tok_idx] == LABEL2ID["OUTDATED"]:
            target_word_indices.add(wid)
    print(f"[DEBUG] Target word indices: {target_word_indices}")
    for idx in target_word_indices:
        term = words[idx]
        base_term = simple_lemmatize(term)
        pos = pos_tags[idx][1]

        # 1. Try to utilize provided_pos_map if present
        pos_simple = None
        if provided_pos_map and base_term in provided_pos_map:
            pos_simple = provided_pos_map[base_term]
            print(f"  [DEBUG] Utilizing provided POS for '{base_term}': '{pos_simple}'")
        else:
            pos_simple = penn_to_simple(pos)
            print(f"  [DEBUG] Falling back to nltk POS for '{base_term}': '{pos_simple}'")

        available_pos = [k[1] for k in OD_TERMS_POS if k[0] == base_term]

        # 2. Replacement logic
        print(f"  [FALLBACK DEBUG] base_term: {base_term}, available_pos: {available_pos}, pos_simple: {pos_simple}, current term: {term}")

        if (base_term, pos_simple) in hardcoded_synonym_map:
            synonym = hardcoded_synonym_map[(base_term, pos_simple)]
            print(f"  [DEBUG] Found synonym for ({base_term}, {pos_simple}): '{synonym}'")
        elif len(available_pos) == 1:
            only_pos = available_pos[0]
            synonym = hardcoded_synonym_map[(base_term, only_pos)]
            print(f"  [FALLBACK] Swapping '{term}' ({base_term}) with '{synonym}' utilizing only available POS '{only_pos}'")
            print(f"  [WARN] NLTK/provided guessed '{pos_simple}' for '{base_term}', but utilizing only available '{only_pos}' from mapping.")
        elif len(available_pos) > 1:
            synonym = term
            print(f"  [WARN] Ambiguous POS for '{base_term}'. NLTK/provided POS '{pos_simple}' not found, options: {available_pos}. No swap.")
        else:
            synonym = term
            print(f"  [WARN] No mapping found for '{base_term}' in synonym map. No swap.")

        # === Pluralization and capitalization logic ===
        is_plural = (pos in ['NNS', 'NNPS'] or
                    (term.endswith('s') and not term.endswith('ss')) or
                    (term.endswith('S') and not term.endswith('SS')))
        is_all_caps = term.isupper()
        possessive_suffix = ""

        # ---- Possessive handling ----
        if term.endswith("'s"):
            possessive_suffix = "'s"
            core_term = term[:-2]
        elif term.endswith("s'"):
            possessive_suffix = "s'"
            core_term = term[:-2]
        else:
            core_term = term

        if synonym != term:
            # Pluralize BEFORE capitalization/possessive
            if is_plural:
                if not p.singular_noun(synonym):  # If not already plural
                    synonym = p.plural(synonym)
            # Capitalize
            if is_all_caps:
                synonym = synonym.upper()
            else:
                synonym = match_capitalization(term, synonym)
            # Now re-add possessive
            synonym = synonym + possessive_suffix
            words[idx] = synonym

    joined = smart_join(words)
    joined = re.sub(r"\s+\.", ".", joined)
    print(f"[DEBUG] Final joined sentence: '{joined}'")
    return joined

# === Predictor With Adversarial Detection and Character/Token Limit ===

def predict(input_text):
    num_tokens = len(custom_tokenize(input_text))
    if len(input_text) > MAX_CHARS:
        err_msg = f"❌ Error: Input exceeds {MAX_CHARS} characters. Returning original input."
        print("Result:", err_msg)
        return input_text
    if num_tokens > MAX_TOKENS:
        err_msg = f"❌ Error: Input exceeds {MAX_TOKENS} tokens. Returning original input."
        print("Result:", err_msg)
        return input_text

    # Only called if limits are NOT exceeded:
    tagged_input = insert_extra_id_tags(input_text)
    tokens = custom_tokenize(input_text)
    pos_tags = nltk.pos_tag(tokens)
    provided_pos_map = {}
    for i, word in enumerate(tokens):
        base = simple_lemmatize(word)
        pos_simple = penn_to_simple(pos_tags[i][1])
        possible_pos = [k[1] for k in hardcoded_synonym_map.keys() if k[0] == base]
        if possible_pos:
            provided_pos_map[base] = possible_pos[0]

    swapped = swap_outdated(tagged_input, model, provided_pos_map=provided_pos_map)
    cleaned = strip_extra_id_tags(swapped)
    log_prediction(input_text, cleaned)
    return cleaned

# === Hardcoded Test Sentence ===

if __name__ == "__main__":
    input_text = "Managers and users' efforts were recognized."
    print("Original:", input_text)
    result = predict(input_text)
    print("Result:", result)
    # # Build the provided_pos_map
    # tokens = custom_tokenize(input_text)
    # pos_tags = nltk.pos_tag(tokens)
    # provided_pos_map = {}
    # for i, word in enumerate(tokens):
    #     base = simple_lemmatize(word)
    #     pos_simple = penn_to_simple(pos_tags[i][1])
    #     # Only add to provided_pos_map if that (base, pos_simple) exists in the synonym map
    #     # Find any pos in the mapping for this base word, utilize the first found
    #     possible_pos = [k[1] for k in hardcoded_synonym_map.keys() if k[0] == base]
    #     if possible_pos:
    #         # Utilize the first, or you could collect all if you want to be more robust
    #         provided_pos_map[base] = possible_pos[0]

    # print(f"[DEBUG] Provided POS map: {provided_pos_map}")
    # # Predict
    # swapped = swap_outdated(tagged_input, model, provided_pos_map=provided_pos_map)
    # # Postprocess: REMOVE all extra_id tags and normalize spaces
    # # result = strip_extra_id_tags(swapped)
    # print("Input:", input_text)
    # print("Result:", swapped) # <-- Print swapped, not result
