"""
mnist_svm.py
~~~~~~~~~

A module to demonstrate baseline classification using Support Vector Machines (SVM) on the MNIST dataset.
"""

from sklearn import svm

from ..data.mnist_loader import load_data


def svm_baseline():
    """
    Train an SVM classifier using the MNIST dataset and evaluate its performance.

    This function loads the MNIST dataset using `mnist_loader.load_data()`, splits it into training and test sets,
    trains an SVM classifier (`svm.SVC`) on the training data, makes predictions on the test data, and prints the
    classification accuracy.

    Prints:
    str: A message displaying the classification accuracy of the SVM on the test set.

    Notes:
    - Uses Support Vector Machine (SVM) with default parameters.
    - Utilizes the `mnist_loader` module to load and preprocess the MNIST dataset.
    """
    # Load MNIST dataset
    training_data, _, test_data = load_data()

    # Prepare training and test data
    X_train, y_train = training_data[0], training_data[1]
    X_test, y_test = test_data[0], test_data[1]

    # Initialize SVM classifier
    clf = svm.SVC(verbose=True)

    # Train SVM classifier
    clf.fit(X_train, y_train)

    # Make predictions on test data
    preds = [int(p) for p in clf.predict(X_test)]

    # Calculate number of correct predictions
    num_correct = sum(int(p == y) for p, y in zip(preds, y_test))

    # Print classification accuracy
    print("Baseline classifier using a SVM: {0} / {1}".format(num_correct, len(y_test)))
