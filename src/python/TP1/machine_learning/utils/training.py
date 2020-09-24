import cv2
import csv
import numpy as np

from TP1.common.common_utils import SUPPORT_VECTOR_MACHINE, DECISION_TREE, create_classifier
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


def train(classifier_name):
    features, labels = load_dataset()

    # decision tree
    classifier = create_classifier(classifier_name)
    classifier.train(features, cv2.ml.ROW_SAMPLE, labels)
    classifier.save('../models/{}.yaml'.format(classifier_name))

    return


if __name__ == '__main__':
    train(SUPPORT_VECTOR_MACHINE)
    # train(DECISION_TREE)


