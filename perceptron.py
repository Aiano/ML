import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    w = None
    w0 = 0
    eta = 1

    def __init__(self):
        self.w = 0
        self.w0 = 0
        pass

    def train(self, data_set, loop_max=10):
        # data_set: array(x: d dimensions, y: scalar)
        d = data_set.shape[1] - 1
        self.w = np.zeros(d)
        self.w0 = 0
        for t in range(loop_max):  # iterate loop
            for data in data_set:
                x = data[:-1]
                y = data[-1]
                if y * (self.w.dot(x) + self.w0) <= 0:
                    self.w = self.w + self.eta * y * x
                    self.w0 = self.w0 + self.eta * y

        return self.w, self.w0

    def predict(self, test_data):
        if test_data.shape[0] != self.w.shape[0]:
            print("Length of test data is wrong.")
            return

        gamma = self.w.dot(test_data) + self.w0
        if gamma > 0:
            return 1
        else:
            return -1


def plot(data_set, w, w0):
    # draw data_set

    # way1:
    # for data in data_set:
    #     x = data[:-1]
    #     y = data[-1]
    #     if y == 1:
    #         plt.scatter(x[0], x[1], c='red', label='+1')
    #     else:
    #         plt.scatter(x[0], x[1], c='blue', label='-1')

    # way2:
    class1_x = [data[0] for data in data_set if data[2] == 1]
    class1_y = [data[1] for data in data_set if data[2] == 1]
    class2_x = [data[0] for data in data_set if data[2] == -1]
    class2_y = [data[1] for data in data_set if data[2] == -1]
    plt.scatter(class1_x, class1_y, c='red', label='+1')
    plt.scatter(class2_x, class2_y, c='blue', label='-1')

    # draw hyperplane
    x = np.linspace(0, 4, 50)
    y = - w[0] / w[1] * x - w0 / w[1]
    line, = plt.plot(x, y, label='hyperplane')

    # add some details
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Perceptron Method')

    # show plot
    plt.show()


if __name__ == '__main__':
    data_set1 = np.array([[3, 3, 1],
                          [4, 3, 1],
                          [1, 1, -1]])
    data_set2 = np.array([[3, 3, 1],
                          [4, 3, 1],
                          [1, 1, -1],
                          [2, 2, -1]])

    perceptron = Perceptron()
    w, w0 = perceptron.train(data_set2, loop_max=50)
    print(w, w0)
    plot(data_set2, w, w0)
