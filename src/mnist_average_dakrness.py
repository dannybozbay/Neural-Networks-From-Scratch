from collections import defaultdict

import mnist_loader


def main():
    training_data, _, test_data = mnist_loader.load_data()
    averages = avg_darkness(training_data)
    num_correct = sum(
        int(predict_digit(image, averages) == digit)
        for image, digit in zip(test_data[0], test_data[1])
    )

    print(
        "Baseline classifier using average darkness of image: {0} / {1}".format(
            num_correct, len(test_data[1])
        )
    )


def avg_darkness(training_data):
    """Return a dictionary whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness
    of training images containing that digit. The dakrness for any
    particular image is just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    averages = defaultdict(float)
    for digit, n in digit_counts.items():
        averages[digit] = darknesses[digit] / n

    return averages


def predict_digit(image, averages):
    """Return the digit whose average darkness in the training data
    is closest to the darkness of the ''image''. Note that 'averages'
    is assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data."""
    darkness = sum(image)
    distances = {
        digit: abs(avg_darkness - darkness) for digit, avg_darkness in averages.items()
    }
    return min(distances, key=distances.get)


if __name__ == "__main__":
    main()
