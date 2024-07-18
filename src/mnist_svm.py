from sklearn import svm

import mnist_loader


def svm_baseline():
    training_data, _, test_data = mnist_loader.load_data()
    X_train, y_train = training_data[0], training_data[1]
    X_test, y_test = test_data[0], test_data[1]

    clf = svm.SVC(verbose=True)
    clf.fit(X_train, y_train)
    preds = [int(p) for p in clf.predict(X_test)]
    num_corect = sum(int(p == y) for p, y in zip(preds, y_test))
    print("Baseline classifier using a SVM: {0} / {1}".format(num_corect, len(y_test)))


if __name__ == "__main__":
    svm_baseline()
