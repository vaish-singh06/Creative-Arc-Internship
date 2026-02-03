from tensorflow.keras import layers

model_layers = [
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Dropout(0.5)
]

print("BatchNorm + Dropout layers created successfully")
