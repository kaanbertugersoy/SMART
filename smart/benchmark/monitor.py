import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def monitor_train_loss(histories, labels=None, save_path=None):
    """
    Plots training and validation loss for multiple model histories with additional features.

    Parameters:
    - histories: A list of history objects, where each object has 'history' with 'loss', 'val_loss', and optionally other metrics.
    - labels: A list of labels corresponding to each model. If None, default labels will be used.
    - save_path: Path to save the plot. If None, the plot is shown interactively.
    """
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(histories))]

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))

    for history, label in zip(histories, labels):
        plt.plot(history.history['loss'],
                 label=f'{label} - Train Loss', linestyle='-', marker='o')
        plt.plot(history.history['val_loss'],
                 label=f'{label} - Val Loss', linestyle='--', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Given Model(s)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
