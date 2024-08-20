from sklearn.preprocessing import StandardScaler


def scale_data(train, test):
    # do not forget mu and std are from "train set", that is important
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# Scaler class expects the same input dimension, but adding prediction
# values on top of the dataframe changes dimension and throws an error
# yet it could not be solved in an easy way

# in any terms scaler function has to return the scaler object


def inverse_scale_data(scaler, data):
    return scaler.inverse_transform(data)
