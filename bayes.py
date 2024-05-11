import os
import argparse
import numpy as np
import heapq
from collections import Counter
from PIL import Image


class Bayes:
    def __init__(self):
        self.alpha = 1
        self.true_data = {}
        self.vector_images_train = {}
        self.vector_images_test = {}
        self.prior_probabilities = {}
        self.feature_probabilities = {}
        self.true_data_file = ''
        self.train_data = ''
        self.test_data = ''
        self.output_file = ''
        self.parse_arguments()
        # loads train and test images and true values
        self.load_data()
        self.get_prior_probabilities()
        self.calculate_feature_probabilities()
        # smoothing factor for laplace smoothing

    def setup_arg_parser(self):
        parser = argparse.ArgumentParser(description='Learn and classify image data.')
        parser.add_argument('train_path', type=str, help='path to the training data directory')
        parser.add_argument('test_path', type=str, help='path to the testing data directory')
        parser.add_argument("-o", metavar='filepath',
                            default='classification.dsv',
                            help="path (including the filename) of the output .dsv file with the results")
        return parser

    def parse_arguments(self) -> None:
        parser = self.setup_arg_parser()
        args = parser.parse_args()
        self.train_data = args.train_path
        self.test_data = args.test_path
        self.output_file = args.o

    def load_image(self, image_path) -> np.ndarray:
        img = np.array(Image.open(image_path)).astype(int).flatten()
        return img

    # loads all the images into memory for easier manipulation
    def load_data(self) -> None:
        for fname_train in os.listdir(self.train_data):
            if fname_train.endswith('.dsv'):
                self.true_data_file = fname_train
                self.read_dsv_file()
            else:
                self.vector_images_train[fname_train] = self.load_image(os.path.join(self.train_data, fname_train))

        for fname_test in os.listdir(self.test_data):
            if not fname_test.endswith('.dsv'):
                self.vector_images_test[fname_test] = self.load_image(os.path.join(self.test_data, fname_test))

    def read_dsv_file(self) -> None:
        with open(os.path.join(self.train_data, self.true_data_file), 'r') as file:
            for line in file:
                str.strip(line)
                key, value = str.strip(line).split(':')
                self.true_data[key] = value

    def get_prior_probabilities(self) -> None:
        found_values = {}
        for fname, _ in self.vector_images_train.items():
            value = self.true_data.get(fname)
            if value is not None:
                if value not in found_values:
                    found_values[value] = 1
                else:
                    found_values[value] += 1

        total_samples = len(self.vector_images_train)
        num_classes = len(found_values)

        for value, count in found_values.items():
            if not (len(self.vector_images_train) == 0):
                self.prior_probabilities[value] = (count + self.alpha) / (total_samples + self.alpha * num_classes)

    def calculate_feature_probabilities(self) -> None:
        epsilon = 10e-3
        for value, _ in self.prior_probabilities.items():
            data = np.array([self.vector_images_train[fname] for fname in self.vector_images_train if self.true_data[fname] == value])
            if data.size == 0:
                continue

            means = np.mean(data, axis=0)
            standard_deviations = np.sqrt(np.var(data, axis=0) + epsilon)
            self.feature_probabilities[value] = (means, standard_deviations)

    def classify_image(self, image: np.ndarray) -> np.ndarray:
        probabilities = {}
        for value_class, (means, standard_devs) in self.feature_probabilities.items():
            probability = self.prior_probabilities[value_class]
            log_base_probability = np.log(probability)
            for i in range(len(image)):
                if standard_devs[i] == 0:
                    continue
                exp = np.exp((-0.5) * (((image[i] - means[i]) / standard_devs[i]) ** 2))
                prob_den_fnc = (1 / (standard_devs[i] * np.sqrt(2 * np.pi))) * exp
                if prob_den_fnc > 0:
                    log_base_probability += np.log(prob_den_fnc)
                else:
                    log_base_probability += float('-inf')

            probabilities[value_class] = log_base_probability

        return max(probabilities, key=probabilities.get)

    def bayes_algorithm(self):
        value_guesses = {}
        for fname, image in self.vector_images_test.items():
            best_guess = self.classify_image(image)
            value_guesses[fname] = best_guess

        self.write_into_dsv(value_guesses)

    def write_into_dsv(self, dictionary):
        with open(self.output_file, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}:{value}\n")


if __name__ == '__main__':
    bayes = Bayes()
    bayes.bayes_algorithm()
