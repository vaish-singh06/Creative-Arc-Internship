from tensorflow.keras import models, layers

def build_rnn():
    model = models.Sequential([
        layers.Embedding(input_dim=1000, output_dim=64),
        layers.LSTM(64),
        layers.Dense(1, activation="sigmoid")
    ])
    return model
