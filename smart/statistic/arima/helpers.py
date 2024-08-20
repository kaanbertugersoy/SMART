
import matplotlib.pyplot as plt


def plot_result(model, data, train, test):
    params = model.get_params()
    d = params['order'][1]

    train_pred = model.predict_in_sample(start=d, end=-1)
    test_pred, conf_int = model.predict(
        n_periods=len(test), return_conf_int=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data, label='Original', color='blue')
    ax.plot(train.index[d:], train_pred, label='Fitted', color='green')
    ax.plot(test.index, test_pred, label='Forecast', color='red')
    ax.fill_between(test.index, conf_int[:, 0],
                    conf_int[:, 1], color='pink', alpha=0.3)
    ax.legend()
    plt.show()


def plot_test(model, test):
    test_pred, conf_int = model.predict(
        n_periods=len(test), return_conf_int=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test, label='True', )
    ax.plot(test.index, test_pred, label='Forecast')
    ax.fill_between(test.index, conf_int[:, 0],
                    conf_int[:, 1], color='pink', alpha=0.3)
    ax.legend()
    plt.show()
