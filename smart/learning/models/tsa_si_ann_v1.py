from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, Input  # type: ignore

INPUT_SHAPE = 0
OUTPUT_SHAPE = 1


# For Regression output activation function is None
# For Binary Classification output activation function is sigmoid
# For Multi-Class Classification output activation function is softmax

# For Regression loss function is mse
# For Binary Classification loss function is binary_crossentropy
# For Multi-Class Classification loss function is categorical_crossentropy


def model_build_fn(input_shape, output_shape):
    inputs = Input(shape=input_shape)

    x = Dense(64)(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)

    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)

    outputs = Dense(output_shape)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
