from matplotlib import pyplot as plt


def plot_digit(digit: list) -> None:
    ''' Take in a list of 64 digits and plot a greyscale image in an 8x8 grid.'''
    im = [digit[index:index+8] for index in range(0,len(digit),8)]
    plt.imshow(im,cmap=plt.cm.gray_r)


def euclidean(a: list, b: list) -> int:
    ''' Take in two lists of coordinate points and return the Euclidean distance
        between them.
    '''
    return sum([(pair[0]-pair[1])**2 for pair in zip(a,b)])**0.5


def sorted_distances(point: list, data: list) -> list:
    ''' Take in an unknown point as a list of coordinates, and a second list
        containing lists of coordinates for every known point. Calculate
        the Euclidean distance between the unknown and every known point
        in the data set. Write the distance and the label of the known
        point as a tuple to a list, and return the list sorted in descending
        order.
    '''
    distances = [(euclidean(point, datum[0]), datum[1]) for datum in data]
    distances.sort(key=lambda x: x[0])
    return distances


def mode(distances: list) -> int:
    ''' Take in a list of tuples containing distances and labels. Return the
        modal label in list.
    '''
    labels = [distance[1] for distance in distances]
    return max(labels, key=labels.count)


def knn(k: int, point: list, data: list) -> int:
    ''' Take in an integer for k, an unkown point as a list of coordinates, and
        a list of lists containing the coordinates for all known points in a
        dataset. Calculate the Euclidean distance between the unknown point and
        every known point, and store the value along with the label of the known
        point in a list. Sort the list by distance in descending order. Return the
        modal label for the k nearest points.
    '''
    distances = sorted_distances(point, data)
    modal_value = mode(distances[:k])
    return modal_value


def train_test(test_split: float, data: list) -> tuple[list, list]:
    ''' Take in a float between 0 and 1 representing a fraction to split a list of
        known data points into two lists of testing data and training data.
    '''
    test_size = int(len(data)*test_split)
    train_data, test_data = data[test_size:], data[:test_size]
    return train_data, test_data


def accuracy(k: int, test_split: float, data: list) -> float:
    ''' Take in a integer for k, a fraction for splittiing the known data points
        into testing and training data, and a list of the known data points. Split
        the data into testing and training data according to the test_split fraction,
        and run the knn function for the given value of k. Count and number of
        correctly classified data points, and return the percentage out of the
        entire dataset.
    '''
    train_data, test_data = train_test(test_split, data)

    success_count = 0
    for point in test_data:
        if knn(k, point[0], train_data) == point[1]:
            success_count += 1

    accuracy = (success_count/len(test_data)) * 100
    return accuracy

