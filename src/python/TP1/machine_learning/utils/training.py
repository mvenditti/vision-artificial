import cv2
import csv
import numpy as np
from TP1.machine_learning.utils.dataset import label_to_int


def load_dataset():
    with open('../dataset/moments.csv', mode='r') as file:
        features = []
        labels = []

        reader = csv.reader(file, delimiter=',')
        for row in reader:
            features.append((row[:7]))
            labels.append(label_to_int(row[7]))

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
    return features, labels


def train():
    features, labels = load_dataset()

    # decision tree
    classifier = cv2.ml.DTrees_create()
    classifier.setCVFolds(1)
    classifier.setMaxDepth(10)
    classifier.train(features, cv2.ml.ROW_SAMPLE, labels)
    classifier.save('tree_shapes_model.yaml')

    # svm
    # classifier = cv2.ml.SVM_create()
    # classifier.setKernel(cv2.ml.SVM_RBF)
    # classifier.train(features, cv2.ml.ROW_SAMPLE, labels)
    # classifier.save('svm_data.dat')

    return


if __name__ == '__main__':
    train()


