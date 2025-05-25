import re

def limit_sentences(text, max_sentences=4, max_chars=50):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > max_sentences or len(text) > max_chars:
        raise ValueError("Condense your input, no more than 4 sentences or 50 characters are allowed.")
    return text
