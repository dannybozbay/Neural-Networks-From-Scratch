"""
plots.py
~~~~~~~


This module provides utility functions for plotting metrics during neural network training, with enhanced visual
appearance using `matplotlib` and `scienceplots`. It includes configurations for creating high-quality plots and 
a function for plotting training, validation, and test metrics over multiple epochs.

Key Features:
- Configures `matplotlib` with the `science` and `high-contrast` styles for consistent and visually distinct plots.
- Customizes plot appearance settings such as grid lines and resolution.
- Defines a function to plot multiple datasets, including training and validation metrics, with options for labeling,
  and y-axis scaling.

Functions:
- plot_metrics: Plots one or more datasets on a single graph with optional labels, custom x and y axis labels, and 
  scaling for accuracy metrics. The function returns a `matplotlib` figure object for further manipulation or saving.

Usage:
- Import the module and call `plot_metrics` with the desired datasets and configuration parameters.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee", "grid", "std-colors"])
# plt.rcParams["figure.dpi"] = 100
# plt.rcParams["savefig.dpi"] = 600
plt.rcParams["legend.fontsize"] = "xx-small"
plt.rcParams["legend.loc"] = "best"


def plot_metrics(
    datasets: Optional[List[List[float]]] = None,
    labels: Optional[List[str]] = None,
    title: str = "",
    xlabel: str = "Epochs",
    ylabel: str = "",
    is_accuracy: bool = False,
) -> plt.Figure:
    """
    Plots multiple datasets on a single graph with optional labeling and customization.

    Parameters:
    - datasets (Optional[List[List[float]]]): A list containing the datasets to be plotted.
      Each dataset should be an iterable of numeric values. Must provide at least one dataset.
    - labels (Optional[List[str]]): A list of labels corresponding to each dataset.
      The number of labels must match the number of datasets. Defaults to None.
    - title (str): The title of the plot. Defaults to an empty string.
    - xlabel (str): The label for the x-axis. Defaults to "Epochs".
    - ylabel (str): The label for the y-axis. If `is_accuracy` is True, this is overridden to "%".
      Defaults to an empty string.
    - is_accuracy (bool): A flag to indicate if the y-axis represents accuracy. If True,
      the y-axis is set to a percentage scale (0-100%) and the plot title is set to "Classification Accuracy".
      Defaults to False.

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object containing the plot.

    Raises:
    - ValueError: If no datasets are provided or if the number of labels does not match the number of datasets.
    """

    # Set default values for datasets and labels if they are None
    datasets = datasets or []
    labels = labels or []

    # Validate that at least one dataset is provided
    if not datasets:
        raise ValueError("A dataset must be provided.")

    # Check if labels are provided and their count matches the number of datasets
    if labels and len(labels) != len(datasets):
        raise ValueError(
            "The number of labels provided must match the number of datasets provided."
        )

    # Create a new figure for the plot
    fig = plt.figure()

    # Determine the maximum length of the datasets
    max_length = max(len(data) for data in datasets)
    x_values = np.arange(
        1, max_length + 1
    )  # X-values representing epochs from 1 to max_length

    # Plot each dataset with its corresponding label and x-values
    for data, label in zip(datasets, labels):
        plt.plot(
            x_values[: len(data)], data, label=label
        )  # Use x_values to align with epoch numbers

    # Adjust plot settings for accuracy metric if applicable
    if is_accuracy:
        plt.title(r"Classification Accuracy (\%)")
        plt.ylim(top=100)
        plt.ylabel(r"%")
    else:
        plt.title(title)
        plt.ylabel(ylabel)

    # Set x-axis properties based on xlabel
    if xlabel == "Epochs":
        plt.xlim(1, max_length)

        # Determine tick interval based on max_length
        if max_length <= 10:
            interval = 1
        elif max_length <= 30:
            interval = 5
        elif max_length <= 50:
            interval = 10
        elif max_length <= 100:
            interval = 20
        elif max_length <= 300:
            interval = 50
        elif max_length <= 500:
            interval = 100
        elif max_length <= 1000:
            interval = 200
        else:
            interval = 100  # Fallback for larger sizes, adjust if needed

        # Set the x-ticks using the determined interval, starting from 1
        ticks = np.arange(0, max_length + 1, interval)
        ticks[0] = 1
        plt.xticks(ticks)

    plt.xlabel(xlabel)
    plt.legend()

    # Return the figure object for further manipulation or saving
    return fig
