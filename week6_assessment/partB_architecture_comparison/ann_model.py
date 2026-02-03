from tensorflow.keras import models, layers

def build_ann(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model
