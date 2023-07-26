import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from PIL import Image
from scipy.interpolate import interp1d
import seaborn as sns
import pickle

from preprocessing import findRotationAngle, rotateImage
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
    image_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text1_rot.png"
    text_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text1.txt"

    # read data
    image = Image.open(image_filename)
    image = np.array(image)
    # remove squigly lines from image
    image = remove_colors(image)
    text = None
    with open(text_filename, "r", encoding="utf-16") as f:
        text = f.readlines()
    text = "".join(text)
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = [x for x in text]

    all_contours = []

    # make border white to account for black image edges
    border_mask = 10
    image[0:border_mask, :] = [255, 255, 255]
    image[:, 0:border_mask] = [255, 255, 255]
    image[image.shape[0] - border_mask:, :] = [255, 255, 255]
    image[:, image.shape[1] - border_mask:] = [255, 255, 255]

    # rotate image back to normal
    rot_angle = findRotationAngle(image)
    image = rotateImage(image, -rot_angle, "nearest")
    plt.imshow(image)
    plt.savefig("text_train_normal_rot.png")
    plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

    print(f"Image for training was rotated {rot_angle * 180.0 / np.pi} degrees")

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
    
    manual_splits = {
        "1-1-0": (6, 7),
        "1-6-0": (6, 7),
        "2-13-2": (6, 6),
        "17-12-0": (5, 6),
    }

    manual_merges = {
        "4-10": (4, 5)
    }

    # for every line
    for line_idx in range(len(line_rects)):
        line = line_rects[line_idx]
        # find every word
        line_image = get_rectangle(image, line)
        threshold = 220
        _, line_image_thresholded = cv2.threshold(line_image, threshold, 255, cv2.THRESH_BINARY)

        # remove white edges
        crop_rect = remove_white_edges(line_image_thresholded) 
        line_image_thresholded = get_rectangle(line_image_thresholded, crop_rect)
        line_image = get_rectangle(line_image, crop_rect)

        projections = line_image_thresholded.mean(axis=0)
        # split on 4 or more white pixels
        word_break_points = [0]
        i = 0
        while i < len(projections) - 4:
            if np.mean(projections[i:i + 5]) > 245:
                word_break_points.append(i)
                word_break_points.append(i + 4)
                i += 4
            else:
                i += 1
        word_break_points.append(projections.shape[0])
        M = line_image.shape[0]
        word_rects = []
        for i in range(0, len(word_break_points) - 1, 2):
            word_rects.append([0, M, word_break_points[i], word_break_points[i + 1]])
        
        # for every word
        for word_idx in range(len(word_rects)):
            word_rect = word_rects[word_idx]
            # need to split letters
            word = get_rectangle(line_image, word_rect)
            word = np.pad(word, 2, "constant", constant_values=(255, 255))
            threshold = 100
            _, word_thresholded = cv2.threshold(word, threshold, 255, cv2.THRESH_BINARY)

            # find all letters in image
            projections = word_thresholded.mean(axis=0)
            letter_break_points = []
            for i in range(len(projections) - 1):
                if projections[i] != projections[i + 1]:
                    if projections[i] == 255:
                        letter_break_points.append(i + 1)
                    if projections[i + 1] == 255:
                        letter_break_points.append(i)
            
            M = word.shape[0]
            letter_rects = []
            for i in range(0, len(letter_break_points) - 1, 2):
                letter_rects.append([0, M, letter_break_points[i], letter_break_points[i + 1]])
            
            new_letter_rects = []
            
            for i in range(len(letter_rects)):
                change = f"{line_idx}-{word_idx}-{i}"
                if change in manual_splits:
                    split = manual_splits[change]
                    start = letter_rects[i][2]
                    end = letter_rects[i][3]
                    rect_1 = [0, M, start, start + split[0]]
                    rect_2 = [0, M, start + split[1], end]
                    new_letter_rects.append(rect_1)
                    new_letter_rects.append(rect_2)
                else:
                    new_letter_rects.append(letter_rects[i])

            letter_rects = new_letter_rects
            
            new_letter_rects = []

            for i in range(len(letter_rects)):
                change = f"{line_idx}-{word_idx}"
                if change in manual_merges:
                    merge = manual_merges[change]
                    start = merge[0]
                    end = merge[1]
                    if i == start:
                        merged = [0, M, letter_rects[i][2], letter_rects[i + 1][3]]
                        new_letter_rects.append(merged)
                    elif i == end:
                        continue
                    else:
                        new_letter_rects.append(letter_rects[i])
                else:
                    new_letter_rects.append(letter_rects[i])


            letter_rects = new_letter_rects

            if line_idx == 17 and word_idx == 7:
                letter_rects.pop(9)

            for letter_idx in range(len(letter_rects)):
                letter_rect = letter_rects[letter_idx]
                letter = get_rectangle(word, letter_rect)
                contours = getcontour(letter)
                all_contours.append(contours)
    
    # construct dataset 
    one_contour = "CEFGHIJKLMNSTUVWXYZcfhklmnrstuvwxyz12357,'.()-"
    two_contour = "ADOPQRabdeijopq4690"
    three_contour = "B8g"

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
            x_t_f = interp1d(np.linspace(0, N_t, len(x_t)) ,x_t)
            y_t_f = interp1d(np.linspace(0, N_t, len(y_t)) ,y_t)

            t = np.arange(N_t)
            x_t = x_t_f(t)
            y_t = y_t_f(t)

            v = []
            for i in range(N_t):
                v.append(complex(x_t[i], y_t[i]))
            fft_v = np.fft.fft(v)
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