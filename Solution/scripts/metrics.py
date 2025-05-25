
import math
import re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(preds, refs):
    if not preds or not refs or not any(preds) or not any(refs):
        return 0.0
    preds = [pred.split() for pred in preds]
    refs = [[ref.split()] for ref in refs]
    return corpus_bleu(refs, preds, smoothing_function=SmoothingFunction().method1)

def basic_whitespace_tokenizer(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()

def compute_transformation_rate(preds, refs, outdated_terms=None):
    if outdated_terms is None:
        raise ValueError("outdated_terms must be provided explicitly.")
    transformed_count = 0
    total_count = 0
    for ref, pred in zip(refs, preds):
        ref_tokens = basic_whitespace_tokenizer(ref)
        pred_tokens = basic_whitespace_tokenizer(pred)
        for r_tok, p_tok in zip(ref_tokens, pred_tokens):
            if r_tok in outdated_terms:
                total_count += 1
                if r_tok != p_tok:
                    transformed_count += 1
    return transformed_count / total_count if total_count > 0 else 0.0
