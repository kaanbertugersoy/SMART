import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def report_scores(analog, comparison_list, plot=False):
    """
    Reports the scores of various metrics comparing the analog (true values) 
    against multiple model predictions.

    Parameters:
    - analog (np.array): The true values to compare against.
    - comparison_list (tuple): Eeach tuple contains a name and the corresponding predictions.
    - plot (optional): If True, generates a bar chart comparing the models based on each metric.

    Example:
    ```python
    report_scores(analog, [('multistep', df.loc[test_idx, 'multistep']), 
                           ('multioutput', df.loc[test_idx, 'multioutput'])])
    ```
    """

    metric_functions = {
        'MAPE': mean_absolute_percentage_error,
        'MSE': mean_squared_error,
        'R2': r2_score
    }

    results = {}

    for metric_name, metric_func in metric_functions.items():
        model_results = {}
        print(f"\nScores for {metric_name}:")
        for name, predictions in comparison_list:
            score = metric_func(analog, predictions)

            print(f"{name}: {score:.4f}")

            model_results[name] = score
        results[metric_name] = model_results

    results_df = pd.DataFrame(results).T

    if plot:
        results_df.plot(kind='bar', figsize=(10, 6))
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.xlabel("Models")
        plt.xticks(rotation=45)
        plt.show()
