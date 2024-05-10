import os
import argparse
import numpy as np
import heapq
from collections import Counter
from PIL import Image


class KNN:
    def __init__(self):
        self.k = 10
        self.true_data = {}
        self.vector_images_train = {}
        self.vector_images_test = {}
        self.true_data_file = ''
        self.train_data = ''
        self.test_data = ''
        self.output_file = ''
        self.parse_arguments()
        self.load_data()

    def setup_arg_parser(self):
        parser = argparse.ArgumentParser(description='Learn and classify image data.')
        parser.add_argument('train_path', type=str, help='path to the training data directory')
        parser.add_argument('test_path', type=str, help='path to the testing data directory')
        parser.add_argument('-k', type=int,
                            help='number of neighbours (if k is 0 the code may decide about proper K by itself')
        parser.add_argument("-o", metavar='filepath',
                            default='classification.dsv',
                            help="path (including the filename) of the output .dsv file with the results")
        return parser

    def parse_arguments(self) -> None:
        parser = self.setup_arg_parser()
        args = parser.parse_args()
        self.train_data = args.train_path
        self.test_data = args.test_path
        self.k = args.k
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

    def write_into_dsv(self, dictionary):
        with open(self.output_file, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}:{value}\n")

    def algorithm(self):
        algorithm_estimates = {}
        for fname_test, image_test in self.vector_images_test.items():
            nearest_neighbors = []
            for fname_train, image_train in self.vector_images_train.items():
                distance = self.get_distance(image_test, image_train)
                heapq.heappush(nearest_neighbors, (-distance, fname_train))
                if len(nearest_neighbors) > self.k:
                    heapq.heappop(nearest_neighbors)
            estimated_values = []
            for _, neighbor in nearest_neighbors:
                estimated_values.append(self.true_data[neighbor])
            algorithm_estimates[fname_test] = self.most_frequent_value(estimated_values)

        self.write_into_dsv(algorithm_estimates)


    def most_frequent_value(self, array):
        counter = Counter(array)
        most_common, _ = counter.most_common(1)[0]
        return most_common

    def get_distance(self, image1, image2):
        diff = image1 - image2
        return np.linalg.norm(diff)




if __name__ == '__main__':
    knn = KNN()
    knn.algorithm()

