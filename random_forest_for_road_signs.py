import albumentations
import collections
import csv
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        flag = True
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(imageio.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
            # to fix bug of dataset
            if c == 33 and int(row[0].split('_')[0]) == 19 and flag:
                flag = False
                images.append(imageio.imread(prefix + row[0]))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSignsTest(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(imageio.imread(prefix + row[0]))  # the 1th column is the filename
        labels.append(row[7])  # the 8th column is the label
    gtFile.close()
    return images, labels


def get_padding_size(image):
    h, w, _ = image.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    return top, bottom, left, right


def padd_images(images):
    result = []
    for index in range(len(images)):
        top, bottom, left, right = get_padding_size(images[index])
        result.append(cv2.copyMakeBorder(images[index],
                                         top, bottom, left, right,
                                         cv2.BORDER_REPLICATE))
    return result


def resize_images(width, heights, images):
    result = []
    for index in range(len(images)):
        result.append(cv2.resize(images[index], (width, heights), interpolation=cv2.INTER_CUBIC))
    return np.asarray(result)


def class_frequency(labels, s=""):
    data = list(map(int, labels))
    c = sorted(collections.Counter(data).items())
    class_num = np.arange(max(data) + 1)
    freq = [i[1] for i in c]

    plt.figure(figsize=(12, 6))
    plt.bar(class_num, freq)
    plt.title("Image class frequency " + s)
    plt.xlabel("Image class")
    plt.ylabel("Frequency")

    plt.show()


def sub_size_plot(measure, sizes, s=''):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(measure, sizes)

    ax.set(xlabel="Image size", ylabel=s,
           title="Image " + s + " dependency on size")
    ax.grid()
    plt.show()


def split_data(train_labels, train_images):
    track = 0
    training_set_labels = np.asarray([])
    training_set_images = None
    validation_set_labels = np.copy(train_labels)
    validation_set_images = np.copy(train_images)

    while track < train_images.shape[0]:
        crystal_ball = random.randint(1, 5)

        if crystal_ball != 1:
            tr_lb = np.split(validation_set_labels, [track, track + 30])
            tr_im = np.split(validation_set_images, [track, track + 30])
            if len(tr_lb[1]) == 0:
                break
            if tr_lb[1][0] in tr_lb[0] or tr_lb[1][0] in tr_lb[2]:
                training_set_labels = np.concatenate((training_set_labels, tr_lb[1]))
                if training_set_images is None:
                    training_set_images = tr_im[1]
                else:
                    training_set_images = np.concatenate((training_set_images, tr_im[1]))
                validation_set_labels = np.concatenate((tr_lb[0], tr_lb[2]))
                validation_set_images = np.concatenate((tr_im[0], tr_im[2]))
                track -= 30
        track += 30

    print('Train set shape:', training_set_labels.shape)
    print('Validation set shape:', validation_set_labels.shape)
    print('Train set', len(training_set_labels) / len(train_labels[:]), '%')
    print('Validation set', len(validation_set_labels) / len(train_labels[:]), '%')

    return training_set_labels, training_set_images, validation_set_labels, validation_set_images


def shuffle_index(labels, images):
    labels_shuf = []
    images_shuf = []
    index_shuf = list(range(len(labels)))
    shuffle(index_shuf)
    for i in index_shuf:
        labels_shuf.append(labels[i])
        images_shuf.append(images[i])
    return np.asarray(labels_shuf), np.asarray(images_shuf)


def augmentation(image):
    image = albumentations.augmentations.transforms.GaussNoise(
        var_limit=(15.0, 50.0), p=0.8)(image=image)['image']
    image = albumentations.augmentations.transforms.Blur(
        blur_limit=2, p=0.3)(image=image)['image']
    image = albumentations.augmentations.transforms.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.5, p=1)(
        image=image)['image']
    image = albumentations.augmentations.transforms.Rotate(
        limit=10, interpolation=1, border_mode=cv2.BORDER_REPLICATE, p=0.2)(
        image=image)['image']
    return image


def augment_data(labels, images):
    data = list(map(int, labels))
    c = sorted(collections.Counter(data).items())
    freq = [i[1] for i in c]
    max_size = max(freq)

    labels_aug = []
    images_aug = []
    high = 0
    low = 0
    for cl in range(0, 43):
        number_in_class = freq[cl]
        high = low + number_in_class
        amount = max_size - number_in_class
        crystal_ball = np.random.randint(0, number_in_class - 1, size=amount)
        for red in range(amount):
            images_aug.append(augmentation(images[low + crystal_ball[red]]))
            labels_aug.append(labels[low + crystal_ball[red]])
        low = high

    labels = np.concatenate((labels, np.asarray(labels_aug)))
    images = np.concatenate((images, np.asarray(images_aug)))

    return labels, images


def normalize_data(images):
    images_norm = []
    for i in range(images.shape[0]):
        images_norm.append(albumentations.augmentations.transforms.Normalize(
            mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0
        )(image=images[i])['image'])
    return np.asarray(images_norm)


def data_as_matrix(images):
    matrix = []
    for im in images:
        matrix.append(im.ravel())
    return np.asarray(matrix)


def plot_confusion_matrix(y_true, y_pred, classes, w,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.get_ylim()
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.set_size_inches(20, 20)
    cl = random.randint(0, 9)
    fig.savefig('conf' + str(w) + ' ' + str(cl) + '.png')
    plt.show()


def plt_show(img):
    print(img.shape)
    imgplot = plt.imshow(img)
    plt.show()


def show_misclass(trueLabels, predictedLabels, images, cut=3):
    wrong = []
    for ind in range(len(trueLabels)):
        if trueLabels[ind] != predictedLabels[ind]:
            wrong.append(ind)

    for w in wrong[:cut]:
        print('True ', trueLabels[w])
        print('Predicted ', predictedLabels[w])
        plt_show(images[w])


def main(w, h, augmentations):
    # path to dataset
    trainImages, trainLabels = readTrafficSigns('GTSRB/Final_Training/Images')
    testImages, testLabels = readTrafficSignsTest('GTSRB/Final_Test/Images')

    # print('Sample from dataset')
    # plt_show(trainImages[47])

    train_images_padded = padd_images(trainImages)
    test_images_padded = padd_images(testImages)

    # print('\nSample of padded')
    # plt_show(train_images_padded[47])

    train_images_resized = resize_images(w, h, train_images_padded)
    test_images_resized = resize_images(w, h, test_images_padded)

    # print('\nSamples of resized')
    # plt_show(train_images_resized[47])
    # plt_show(train_images_resized[695])
    # plt_show(train_images_resized[13874])

    # print('\nSamples of augmented')
    # plt_show(augmentation(train_images_resized[47]))
    # plt_show(augmentation(train_images_resized[695]))
    # plt_show(augmentation(train_images_resized[13874]))

    # split data 80/20
    training_set_labels, training_set_images, validation_set_labels, validation_set_images = split_data(trainLabels,
                                                                                                        train_images_resized)

    # # plot frequency plot before and after split
    # class_frequency(trainLabels, 'before split')
    # class_frequency(training_set_labels, 'after split')

    if augmentations:
        # augment data
        training_set_labels, training_set_images = augment_data(training_set_labels, training_set_images)

    # # plot frequency plot after augmentations
    # class_frequency(training_set_labels, 'after augmentation')

    # shuffle sets
    training_set_labels, training_set_images = shuffle_index(training_set_labels, training_set_images)
    validation_set_labels, validation_set_images = shuffle_index(validation_set_labels, validation_set_images)

    # normalizing images
    training_set_images = normalize_data(training_set_images)
    validation_set_images = normalize_data(validation_set_images)
    test_set_images = normalize_data(test_images_resized)

    # flattaring and combing into 2D matrix
    train_mat = data_as_matrix(training_set_images)
    valid_mat = data_as_matrix(validation_set_images)
    test_mat = data_as_matrix(test_set_images)

    # Random Forest Classifier
    rfc1 = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=30, min_samples_split=7,
                                  bootstrap=False, n_jobs=-1, random_state=42, verbose=1, warm_start=True)
    start_time = time.time()
    rfc1.fit(train_mat, training_set_labels)
    finish_time = time.time() - start_time

    predicted_set_labels_train = rfc1.predict(train_mat)
    predicted_set_labels_valid = rfc1.predict(valid_mat)
    predicted_set_labels_test = rfc1.predict(test_mat)

    # accuracy score
    print('\nTrain accuracy: ', accuracy_score(training_set_labels, predicted_set_labels_train))
    test_accuracy = accuracy_score(testLabels, predicted_set_labels_test)
    print('Test accuracy: ', test_accuracy)
    print('Validation accuracy: ', accuracy_score(validation_set_labels, predicted_set_labels_valid))

    # plot_confusion_matrix(validation_set_labels, predicted_set_labels_valid, np.arange(43), w)

    show_misclass(testLabels, predicted_set_labels_test, test_set_images)

    # metrics per class for test
    print('\nMetrics per class for test set\n', classification_report(np.asarray(testLabels).astype(np.int),
                                                                      predicted_set_labels_test.astype(np.int)))

    return finish_time, test_accuracy


if __name__ == "__main__":
    finish_time = []
    accuracy = []
    sizes = [5, 15, 30, 60, 80, 100]

    print('Size 30*30 without augmentations')
    ft, ac = main(30, 30, augmentations=False)

    for s in sizes:
        print('-----------------------------------------------------')
        print('\nSize', s, '*', s, ' with augmentations')
        ft, ac = main(s, s, augmentations=True)
        finish_time.append(ft)
        accuracy.append(ac)

    sub_size_plot(finish_time, sizes, 'time to fit')
    sub_size_plot(accuracy, sizes, 'test accuracy')