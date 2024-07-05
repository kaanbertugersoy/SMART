

def scale_data(train, test):
    # do not forget mu and std are from "train set", that is important
    mu, std = train.mean(), train.std()
    train_s = (train - mu) / std
    test_s = (test - mu) / std
    return train_s, test_s
