import tensorflow as tf

def transformer_block(x, num_heads, ff_dim, dropout=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1]
    )(x, x)
    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)

    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

def build_audio_transformer(
    input_shape=(400, 80),  # time, n_mels
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=4
):
    """
    Audio Transformer for log-mel spectrograms.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Patch embedding (simple dense projection)
    x = tf.keras.layers.Dense(embed_dim)(inputs)

    for _ in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="audio_transformer")
    return model