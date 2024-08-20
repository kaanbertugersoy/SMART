from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

model_config = {
    'lstm_spec_seq': False
}


class NNModel:

    def model_build_fn(number_of_tails, tail_input_shape, output_shape):
        inputs = []
        tails = []

        for _ in range(number_of_tails):
            i = Input(shape=tail_input_shape)
            x = Dense(64, activation='relu')(i)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)

            inputs.append(i)
            tails.append(x)

        x = Concatenate()(tails)
        x = Dense(output_shape)(x)

        model = Model(inputs=inputs, outputs=x)
        return model
