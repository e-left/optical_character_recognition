import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from PIL import Image
from scipy.interpolate import interp1d
import seaborn as sns
import pickle

from contours import get_rectangle, getcontour

def remove_colors(x):
    M = x.shape[0]
    N = x.shape[1]
    for x_t in range(M):
        for y_t in range(N):
            r = x[x_t][y_t][0]
            g = x[x_t][y_t][1]
            b = x[x_t][y_t][2]

            if r != g or g != b or r != b:
                x[x_t][y_t][0:3] = [255, 255, 255]
    return x

def remove_white_edges(x):
    line_image_proj_vertical = x.mean(axis=0)
    line_image_proj_horizontal = x.mean(axis=1)
    x_start = 0
    while x_start < len(line_image_proj_horizontal) and line_image_proj_horizontal[x_start] == 255:
        x_start += 1
    x_end = line_image_proj_horizontal.shape[0] - 1
    while x_end > 0 and line_image_proj_horizontal[x_end] == 255:
        x_end -= 1
    y_start = 0
    while y_start < len(line_image_proj_vertical) and line_image_proj_vertical[y_start] == 255:
        y_start += 1
    y_end = line_image_proj_vertical.shape[0] - 1
    while y_end > 0 and line_image_proj_vertical[y_end] == 255:
        y_end -= 1

    return [x_start, x_end, y_start, y_end]


def create_dataset(filename, N1, N2, N3):
    image_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text1_v3.png"
    text_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text1_v3.txt"

    # read data
    image = Image.open(image_filename)
    image = np.array(image)
    # remove blue frame from image
    image = remove_colors(image)
    text = None
    with open(text_filename, "r") as f:
        text = f.readlines()
    text = "".join(text)
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = [x for x in text]

    # rotate image back to normal
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

    # pad image with 10 pixels
    image = np.pad(image, 10, "constant", constant_values=(255, 255))

    all_contours = []

    # split lines
    threshold = 150
    _, image_thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # find all lines in image
    projections = image_thresholded.mean(axis=1)
    line_break_points = []
    for i in range(len(projections) - 1):
        if projections[i] != projections[i + 1]:
            if projections[i] == 255:
                line_break_points.append(i + 1)
            if projections[i + 1] == 255:
                line_break_points.append(i)
    
    N = image_thresholded.shape[1]

    line_rects = []
    for i in range(0, len(line_break_points) - 1, 2):
        line_rects.append([line_break_points[i], line_break_points[i + 1], 0, N])
    
    for line_idx in range(len(line_rects)):
        line_rect = line_rects[line_idx]
        line_image = get_rectangle(image, line_rect)
        text_line = []
        # split words
        threshold = 220
        _, line_image_thresholded = cv2.threshold(line_image, threshold, 255, cv2.THRESH_BINARY)

        # remove white edges
        crop_rect = remove_white_edges(line_image_thresholded) 
        line_image_thresholded = get_rectangle(line_image_thresholded, crop_rect)
        line_image = get_rectangle(line_image, crop_rect)
        line_image_blurred = cv2.GaussianBlur(line_image, (15, 15), 0)

        projections = line_image_blurred.mean(axis=0)
        word_break_points = [0]

        for i in range(len(projections) - 1):
            if projections[i] == projections[i + 1]:
                continue 
            if projections[i] == 255:
                word_break_points.append(i + 1)
            if projections[i + 1] == 255:
                word_break_points.append(i)

        word_break_points.append(projections.shape[0])
        M = line_image.shape[0]
        word_rects = []
        for i in range(0, len(word_break_points) - 1, 2):
            word_rects.append([0, M, word_break_points[i], word_break_points[i + 1]])
        
        for word_idx in range(len(word_rects)):
            word_rect = word_rects[word_idx]
            word = get_rectangle(line_image, word_rect)

            word = np.pad(word, 1, "constant", constant_values=(255, 255))

            # we have every word, split letters, determine which one it is and append to line
            threshold = 100
            _, word_thresholded = cv2.threshold(word, threshold, 255, cv2.THRESH_BINARY)
            letter_break_points = []
            projections = word_thresholded.mean(axis=0)
            for i in range(len(projections) - 1):
                if projections[i] == projections[i + 1]:
                    continue 
                if projections[i] == 255:
                    letter_break_points.append(i + 1)
                if projections[i + 1] == 255:
                    letter_break_points.append(i)

            M = word.shape[0]
            letter_rects = []
            for i in range(0, len(letter_break_points) - 1, 2):
                letter_rects.append([0, M, letter_break_points[i], letter_break_points[i + 1]])
            
            for letter_idx in range(len(letter_rects)):
                letter_rect = letter_rects[letter_idx]
                letter = get_rectangle(word, letter_rect)
                letter = get_rectangle(letter, remove_white_edges(letter))
                contours = getcontour(letter) 
                all_contours.append(contours)
    
    # construct dataset 
    one_contour = "CEFGHIJKLMNSTUVWXYZcfhklmnrstuvwxyz12357,'.()-"
    two_contour = "ADOPQRabdegijopq469;"
    three_contour = "B80"

    datalen = len(text)
    dataset = []
    for i in range(datalen):
        dataset.append([text[i], all_contours[i]])
    
    inconsistent_entries = []
    for i in range(datalen):
        letter = dataset[i][0]
        num_of_contours = len(dataset[i][1])
        target_num_of_contours = 0
        if letter in one_contour:
            target_num_of_contours = 1
        elif letter in two_contour:
            target_num_of_contours = 2
        elif letter in three_contour:
            target_num_of_contours = 3
        if num_of_contours != target_num_of_contours:
            print(f"Inconsisten entry: letter {letter}, found {num_of_contours}, actual {target_num_of_contours}")
            inconsistent_entries.append(i)

    new_dataset = []
    for i in range(datalen):
        if i not in inconsistent_entries:
            new_dataset.append(dataset[i])
    
    print(f"Removed {len(inconsistent_entries)} inconsistent letters")

    
    dataset = new_dataset
    datalen = len(dataset)

    new_dataset = []

    for entry in dataset:
        letter = entry[0]
        fft_contours = []
        contours = entry[1]
        N = [N1, N2, N3]
        N_t = N[len(contours) - 1]
        for i in range(len(contours)):
            contour = contours[i]
            x_t = []
            y_t = []
            for point in contour:
                x_t.append(point[0])
                y_t.append(point[1])
            x_t_f = interp1d(np.arange(0, len(x_t)), x_t)
            y_t_f = interp1d(np.arange(0, len(y_t)), y_t)

            t_x = np.linspace(0, len(x_t) - 1, N_t, endpoint=False)
            t_y = np.linspace(0, len(y_t) - 1, N_t, endpoint=False)
            x_t = x_t_f(t_x)
            y_t = y_t_f(t_y)

            v = []
            for i in range(N_t):
                v.append(complex(x_t[i], y_t[i]))
            fft_v = np.fft.fft(v)
            fft_v = np.fft.fftshift(fft_v)
            fft_v_mag = np.abs(fft_v)
            fft_v_mag_final = fft_v_mag[1:]
            fft_contours.append(fft_v_mag_final)
        new_dataset.append([letter, fft_contours])

        dataset = new_dataset

    with open(filename, "wb") as f:
        pickle.dump(dataset, f)
        print(f"Successfully created dataset with N1 = {N1}, N2 = {N2}, N3 = {N3}") 

def train_knn_model(dataset_filename, k1, k2, k3, display_metrics):
    # load dataset
    dataset = None
    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)
        print("Successfuly loaded dataset")
    
    # split to three classes depending on contour num
    dataset_1 = []
    dataset_2 = []
    dataset_3 = []

    for data_entry in dataset:
        if len(data_entry[1]) == 1:
            dataset_1.append(data_entry)
        elif len(data_entry[1]) == 2:
            dataset_2.append(data_entry)
        elif len(data_entry[1]) == 3:
            dataset_3.append(data_entry)
    
    print(f"Out of {len(dataset)} entries, {len(dataset_1)} have 1 contour, {len(dataset_2)} have 2 contous and {len(dataset_3)} have 3 contours.")

    # perform train-test split, but make sure to include every class in both train and test
    # split class one
    one_contour_counts = {}
    for i in range(len(dataset_1)):
        entry = dataset_1[i]
        letter = entry[0]
        if letter in one_contour_counts:
            one_contour_counts[letter].append(i)
        else:
            one_contour_counts[letter] = [i]
    
    one_train_idxs = []
    for letter in one_contour_counts:
        letter_count = len(one_contour_counts[letter])
        if letter_count == 1:
            print(f"letter {letter} has only 1 occurence")
        split_idx = round(letter_count * 0.7)
        indexes = one_contour_counts[letter]
        train_indexes = indexes[:split_idx]
        one_train_idxs.extend(train_indexes)
    
    train_one = []
    test_one = []

    for i in range(len(dataset_1)):
        entry = dataset_1[i]
        if i in one_train_idxs:
            train_one.append(entry)
        else:
            test_one.append(entry)
    
    # split class two
    two_contour_counts = {}
    for i in range(len(dataset_2)):
        entry = dataset_2[i]
        letter = entry[0]
        if letter in two_contour_counts:
            two_contour_counts[letter].append(i)
        else:
            two_contour_counts[letter] = [i]
    
    two_train_idxs = []
    for letter in two_contour_counts:
        letter_count = len(two_contour_counts[letter])
        if letter_count == 1:
            print(f"letter {letter} has only 1 occurence")
        split_idx = round(letter_count * 0.7)
        indexes = two_contour_counts[letter]
        train_indexes = indexes[:split_idx]
        two_train_idxs.extend(train_indexes)
    
    train_two = []
    test_two = []

    for i in range(len(dataset_2)):
        entry = dataset_2[i]
        if i in two_train_idxs:
            train_two.append(entry)
        else:
            test_two.append(entry)

    # split class three
    three_contour_counts = {}
    for i in range(len(dataset_3)):
        entry = dataset_3[i]
        letter = entry[0]
        if letter in three_contour_counts:
            three_contour_counts[letter].append(i)
        else:
            three_contour_counts[letter] = [i]
    
    three_train_idxs = []
    for letter in three_contour_counts:
        letter_count = len(three_contour_counts[letter])
        if letter_count == 1:
            print(f"letter {letter} has only 1 occurence")
        split_idx = round(letter_count * 0.7)
        indexes = three_contour_counts[letter]
        train_indexes = indexes[:split_idx]
        three_train_idxs.extend(train_indexes)
    
    train_three = []
    test_three = []

    for i in range(len(dataset_3)):
        entry = dataset_3[i]
        if i in three_train_idxs:
            train_three.append(entry)
        else:
            test_three.append(entry)

    print(f"in total we have {len(train_one)} training data and {len(test_one)} testing data (total {len(dataset_1)}) for 1 contour")
    print(f"in total we have {len(train_two)} training data and {len(test_two)} testing data (total {len(dataset_2)}) for 2 contours")
    print(f"in total we have {len(train_three)} training data and {len(test_three)} testing data (total {len(dataset_3)}) for 3 contours")

    classifier_one = KNeighborsClassifier(k1)
    classifier_two = KNeighborsClassifier(k2)
    classifier_three = KNeighborsClassifier(k3)

    # split x and y
    # flatten x 
    train_one_x = []
    train_one_y = []
    test_one_x = []
    test_one_y = []
    for entry in train_one:
        train_one_x.append(np.array(entry[1]).flatten())
        train_one_y.append(entry[0])
    for entry in test_one:
        test_one_x.append(np.array(entry[1]).flatten())
        test_one_y.append(entry[0])

    train_two_x = []
    train_two_y = []
    test_two_x = []
    test_two_y = []
    for entry in train_two:
        train_two_x.append(np.array(entry[1]).flatten())
        train_two_y.append(entry[0])
    for entry in test_two:
        test_two_x.append(np.array(entry[1]).flatten())
        test_two_y.append(entry[0])

    train_three_x = []
    train_three_y = []
    test_three_x = []
    test_three_y = []
    for entry in train_three:
        train_three_x.append(np.array(entry[1]).flatten())
        train_three_y.append(entry[0])
    for entry in test_three:
        test_three_x.append(np.array(entry[1]).flatten())
        test_three_y.append(entry[0])

    classifier_one.fit(train_one_x, train_one_y)
    classifier_two.fit(train_two_x, train_two_y)
    classifier_three.fit(train_three_x, train_three_y)

    print(f"Trained classifiers on test set with k1 = {k1}, k2 = {k2}, k3 = {k3}")

    if display_metrics:

        pred_1 = classifier_one.predict(test_one_x)
        pred_2 = classifier_two.predict(test_two_x)
        pred_3 = classifier_three.predict(test_three_x)

        y_actual = []
        y_actual.extend(test_one_y)
        y_actual.extend(test_two_y)
        y_actual.extend(test_three_y)

        y_pred = []
        y_pred.extend(pred_1)
        y_pred.extend(pred_2)
        y_pred.extend(pred_3)

        conf_matrix = confusion_matrix(y_actual, y_pred)
        classes = sorted(list(set(y_actual)))

        sns.heatmap(conf_matrix, xticklabels=classes, yticklabels=classes)
        plt.savefig("conf_matrix_train_test.png")
        plt.show()
        weighted_acc = balanced_accuracy_score(y_actual, y_pred)
        print(f"For training validation, weighted accuracy is {weighted_acc}")

    return (classifier_one, classifier_two, classifier_three)