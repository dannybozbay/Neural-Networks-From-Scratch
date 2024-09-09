import sys

sys.path.append("..")

from src.baselines import mnist_avg_darkness
from src.data import mnist_loader


def main():
    """
    Main function to load MNIST data, compute average darkness for each digit,
    and evaluate a baseline classifier based on average darkness.

    Prints the accuracy of the baseline classifier using average darkness.
    """
    training_data, _, test_data = mnist_loader.load_data()
    averages = mnist_avg_darkness.avg_darkness(training_data)
    num_correct = sum(
        int(mnist_avg_darkness.predict_digit(image, averages) == digit)
        for image, digit in zip(test_data[0], test_data[1])
    )

    print(
        "Baseline classifier using average darkness of image: {0} / {1}".format(
            num_correct, len(test_data[1])
        )
    )


if __name__ == "__main__":
    main()
