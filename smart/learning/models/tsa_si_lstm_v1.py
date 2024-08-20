from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Input, GlobalMaxPooling1D, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

model_config = {
    'lstm_spec_seq': True
}

INPUT_SHAPE = 0
OUTPUT_SHAPE = 1


class NNModel:

    def model_build_fn(input_shape, output_shape):
        i = Input(shape=input_shape)

        x = LSTM(64, return_sequences=True)(i)
        x = LSTM(64, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

        o = Dense(output_shape)(x)

        model = Model(inputs=i, outputs=o)

        return model
