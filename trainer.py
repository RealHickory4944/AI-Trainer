import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --- 1. CONFIGURATION ---
DATA_PATH = 'WikipediaSimpleEnglish.txt' 
LOAD_PATH = 'weights.weights.h5'
SAVE_PATH = 'weights.weights.h5'
SEQ_LEN = 128
BATCH_SIZE = 512
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
EPOCHS = 10  # Set how many MORE epochs you want to run

# --- 2. ACCELERATOR SETUP ---
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

# --- 3. DATA PREPROCESSING ---
def prepare_dataset(file_path, seq_len, batch_size):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("this is a sample text file to train your llm model " * 500)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset, len(vocab), char2idx, idx2char

dataset, vocab_size, char2idx, idx2char = prepare_dataset(DATA_PATH, SEQ_LEN, BATCH_SIZE)

# --- 4. CUSTOM LAYERS ---

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)
        self.sequence_length = sequence_length

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        
        mask = tf.range(seq_len)[:, None] >= tf.range(seq_len)
        mask = tf.cast(mask, dtype=tf.bool)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        mask = tf.tile(mask, [batch_size, 1, 1])

        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- 5. MODEL BUILDING & TRAINING ---

with strategy.scope():
    inputs = layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    x = PositionalEmbedding(SEQ_LEN, vocab_size, D_MODEL)(inputs)
    
    for _ in range(NUM_LAYERS):
        x = TransformerBlock(D_MODEL, NUM_HEADS, D_MODEL * 4)(x)

    outputs = layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

# RESUME LOGIC
if os.path.exists(LOAD_PATH):
    print(f"\n[INFO] Found existing weights at {LOAD_PATH}. Loading to resume training...")
    model.load_weights(LOAD_PATH)
else:
    print("\n[INFO] No existing weights found. Starting training from scratch.")

model.summary()

# Training
model.fit(dataset, epochs=EPOCHS)

# Save once after training is complete
print(f"\n[INFO] Training session complete. Saving updated weights to {SAVE_PATH}...")
model.save_weights(SAVE_PATH)

# --- 6. INFERENCE ---

def generate_text(model, start_str, length=100, temperature=1.0):
    try:
        input_eval = [char2idx[s] for s in start_str]
    except KeyError as e:
        return f"Error: Character {e} not found in vocab."
        
    input_eval = tf.expand_dims(input_eval, 0)
    generated = []

    for _ in range(length):
        if input_eval.shape[1] > SEQ_LEN:
            curr_input = input_eval[:, -SEQ_LEN:]
        else:
            pad_len = SEQ_LEN - input_eval.shape[1]
            curr_input = tf.pad(input_eval, [[0, 0], [pad_len, 0]])

        predictions = model(curr_input, training=False)
        predictions = predictions[:, -1, :] 
        
        predictions = predictions / max(temperature, 1e-7)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        
        input_eval = tf.concat([input_eval, [[predicted_id]]], axis=1)
        generated.append(idx2char[predicted_id])
    
    return start_str + "".join(generated)

print("\n--- SAMPLE GENERATION ---")
print(generate_text(model, start_str="the ", length=200, temperature=0.8))
