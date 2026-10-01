import pickle

import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

try:
    import pydot  # noqa: F401
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None

# Example training data
input_texts = ["hi", "hello", "yes", "no", "go", "stop"]
target_texts = ["salut", "bonjour", "oui", "non", "va", "arrêt"]

# Create vocabulary sets
input_characters = sorted({ch for text in input_texts for ch in text})
target_characters = sorted({ch for text in target_texts for ch in text})

# Add padding and start/end tokens for decoder
input_characters = ["<pad>"] + input_characters
target_characters = ["<pad>", "<start>", "<end>"] + target_characters

input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}

# Sequence lengths
max_input_len = max(len(text) for text in input_texts)
max_target_len = max(len(text) for text in target_texts) + 2

# Prepare encoder input data
encoder_input_data = np.zeros((len(input_texts), max_input_len), dtype="int32")
for i, text in enumerate(input_texts):
    for t, char in enumerate(text):
        encoder_input_data[i, t] = input_token_index[char]

# Prepare decoder input and target data
num_decoder_tokens = len(target_characters)
decoder_input_data = np.zeros((len(target_texts), max_target_len), dtype="int32")
decoder_target_data = np.zeros((len(target_texts), max_target_len, num_decoder_tokens), dtype="float32")

for i, text in enumerate(target_texts):
    decoder_input_data[i, 0] = target_token_index["<start>"]
    for t, char in enumerate(text):
        decoder_input_data[i, t + 1] = target_token_index[char]
        decoder_target_data[i, t + 1, target_token_index[char]] = 1.0
    decoder_target_data[i, len(text) + 1, target_token_index["<end>"]] = 1.0

# Model parameters
latent_dim = 64

# Encoder model
encoder_inputs = Input(shape=(max_input_len,), dtype="int32", name="encoder_input")
encoder_embedding = Embedding(len(input_characters), latent_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder model
decoder_inputs = Input(shape=(max_target_len,), dtype="int32", name="decoder_input")
decoder_embedding = Embedding(len(target_characters), latent_dim, mask_zero=True)(decoder_inputs)
decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(
    decoder_embedding, initial_state=encoder_states
)
decoder_dense = Dense(len(target_characters), activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Build and compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save metadata
with open("seq2seq_metadata.pkl", "wb") as file:
    pickle.dump(
        {
            "input_token_index": input_token_index,
            "target_token_index": target_token_index,
            "max_input_len": max_input_len,
            "max_target_len": max_target_len,
        },
        file,
    )

# Display model structure
model.summary()
if plot_model is not None:
    try:
        plot_model(model, to_file="seq2seq_model.png", show_shapes=True)
        print("Model diagram saved to seq2seq_model.png")
    except (ImportError, OSError, FileNotFoundError, RuntimeError) as exc:
        print(f"Skipping model diagram generation: {exc}")
else:
    print("pydot/graphviz is not installed; skipping model diagram generation.")

print("\nModel built successfully.")
print("Encoder input shape:", encoder_input_data.shape)
print("Decoder input shape:", decoder_input_data.shape)
print("Decoder target shape:", decoder_target_data.shape)