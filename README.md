# English-to-French Translator

An interactive command-line translator powered by the pretrained [Helsinki-NLP OPUS-MT English-to-French model](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr). It uses PyTorch and Hugging Face Transformers for local inference.

## Requirements

- Python
- Internet access for the first model download (about 300 MB)

The model files are cached locally after the first download. Translation runs locally and does not send entered text to a translation API.

## Setup and Run

Clone the repository and enter its directory:

```bash
git clone https://github.com/ARNAV29042005/Language-Translating.git
cd Language-Translating
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\seq2seq_model.py
```

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python seq2seq_model.py
```

The first launch downloads the model. When the `English>` prompt appears, enter text to translate. Type `quit` to exit.

```text
English> Yesterday it was raining.
French> Hier, il pleuvait.
```

## How It Works

The tokenizer converts English text to model tokens, the pretrained sequence-to-sequence model generates French tokens, and the tokenizer converts the result back to text. The script uses a CUDA GPU when available and otherwise runs on the CPU. Longer inputs are divided into sentence- and token-sized chunks to fit the model's context limit.

## Scope and Limitations

- Supports English-to-French translation only.
- Uses a pretrained model; this project does not train or fine-tune it.
- Long input is translated in chunks, so context between chunks may be lost.
- Translation quality has not been formally benchmarked and may vary.