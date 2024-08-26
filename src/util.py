"""
util.py
~~~~~~~

This module provides utility functions for plotting metrics during the training of neural networks. It includes
configurations for matplotlib to enhance the visual appearance of plots and a function to plot training, validation, 
and test metrics over multiple epochs.

Key Features:
- Configures matplotlib styles and parameters for consistent and high-quality plots.
- Defines a function to plot training, validation, and test accuracy or other metrics over epochs.

Functions:
- plot_metrics: Plots the provided training, validation, and test metrics over epochs and returns the plot as a figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 1000
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["legend.loc"] = "best"


def plot_metrics(
    training_data=[],
    validation_data=[],
    test_data=[],
    metric_name="Accuracy",
    ylabel="%",
):
    """
    Plots training, validation, and test metrics over epochs.

    Parameters:
    - training_data (list of float): List of metric values for the training dataset, indexed by epoch.
    - validation_data (list of float): List of metric values for the validation dataset, indexed by epoch.
    - test_data (list of float): List of metric values for the test dataset, indexed by epoch.
    - metric_name (str): The name of the metric to display in the plot title. Default is "Accuracy".
    - ylabel (str): The label for the y-axis. Default is "%".

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.

    Raises:
    - ValueError: If none of the training_data, validation_data, or test_data are provided.
    """

    # Check if at least one dataset is provided, raise an error if none are.
    if not training_data and not validation_data and not test_data:
        raise ValueError(
            "At least one of training_data, validation_data, or test_data must be provided."
        )

    # Dictionary to hold the data sets with their corresponding labels
    data_sets = {
        "Training Data": training_data,
        "Validation Data": validation_data,
        "Test Data": test_data,
    }

    # Create a new figure for the plot
    fig = plt.figure()

    # Iterate over the data sets and plot each if it is provided
    for label, data in data_sets.items():
        if data:  # Plot only if the data list is not empty
            plt.plot(data, label=label)

    # Set the title, labels, and other plot configurations
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    # Determine the maximum length of the data lists for setting x-axis ticks and limits
    max_len = max(len(training_data), len(validation_data), len(test_data))

    # Calculate the interval for x-axis ticks, aiming for approximately 7 ticks
    tick_interval = max(1, max_len // 6)  # Ensure interval is at least 1

    # Generate x-ticks and set the x-axis limits
    xticks = np.arange(0, max_len + 1, tick_interval)
    plt.xticks(xticks)
    plt.xlim(0, max_len)

    # If the metric is "Accuracy", set the y-axis limit to 100%
    if metric_name == "Accuracy":
        plt.ylim(top=100)

    # Add a legend to the plot
    plt.legend()

    # Return the figure object for further manipulation or saving
    return fig
