from tensorflow.keras.models import Model  # type: ignore
from tesorflow.keras.layers import Dense, Add, Input  # type: ignore


def model_build_fn(input_shape, output_shape, num_blocks=3, hidden_units=128):

    inputs = Input(shape=input_shape)
    x = inputs

    # Create N-BEATS blocks
    for _ in range(num_blocks):
        # Trend Block
        trend_block = Dense(hidden_units, activation='relu')(x)
        trend_block = Dense(hidden_units, activation='relu')(trend_block)
        trend_output = Dense(output_shape[0])(trend_block)

        # Seasonal Block
        seasonal_block = Dense(hidden_units, activation='relu')(x)
        seasonal_block = Dense(hidden_units, activation='relu')(seasonal_block)
        seasonal_output = Dense(output_shape[0])(seasonal_block)

        # Combine outputs from trend and seasonal blocks
        x = Add()([trend_output, seasonal_output])

    # Final output layer
    output = Dense(output_shape[0])(x)

    model = Model(inputs=inputs, outputs=output)

    return model
