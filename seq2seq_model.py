import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


def _token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_sentence(sentence, tokenizer, max_input_tokens):
    words = sentence.split()
    parts = []
    current_words = []

    for word in words:
        if _token_count(word, tokenizer) > max_input_tokens:
            if current_words:
                parts.append(" ".join(current_words))
                current_words = []
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            for start in range(0, len(word_tokens), max_input_tokens):
                token_chunk = word_tokens[start : start + max_input_tokens]
                parts.append(tokenizer.decode(token_chunk, skip_special_tokens=True))
            continue

        candidate = " ".join(current_words + [word])
        if current_words and _token_count(candidate, tokenizer) > max_input_tokens:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        parts.append(" ".join(current_words))
    return parts


def _split_text(text, tokenizer, max_input_tokens):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        chunks.extend(_split_long_sentence(sentence, tokenizer, max_input_tokens))
    return chunks


def main():
    print(f"Loading {MODEL_NAME}. The first run downloads the model files.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_positions = getattr(model.config, "max_position_embeddings", 512)
    max_input_tokens = max(16, max_positions - 8)

    def translate(text):
        if not text.strip():
            raise ValueError("Enter some English text to translate.")

        translated_chunks = []
        for chunk in _split_text(text, tokenizer, max_input_tokens):
            encoded = tokenizer(chunk, return_tensors="pt", truncation=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    num_beams=4,
                    max_length=max_input_tokens + 1,
                )
            translated_chunks.append(tokenizer.decode(output[0], skip_special_tokens=True))
        return " ".join(translated_chunks)

    print("English-to-French translator ready. Type 'quit' to exit.")
    while True:
        try:
            text = input("English> ")
        except EOFError:
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        try:
            print(f"French> {translate(text)}")
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()