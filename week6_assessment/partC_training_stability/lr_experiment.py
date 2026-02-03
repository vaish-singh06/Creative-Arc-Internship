import tensorflow as tf

optimizer_high = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer_normal = tf.keras.optimizers.Adam(learning_rate=0.001)

print("Learning rate optimizers created successfully")
