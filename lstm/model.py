import tensorflow as tf
from tensorflow.keras import layers

def build_model():

    model = tf.keras.models.Sequential([
        layers.LSTM(64, input_shape=(8,4)),
        layers.Dense(24),            # 24 because 12 future timesteps * 2 coordinate (x,y)
        layers.Reshape((12,2))
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer='adam',
        loss=loss_fn, 
        metrics = ['mae']
    )

    return model