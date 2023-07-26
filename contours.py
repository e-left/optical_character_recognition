import numpy as np
from functools import cmp_to_key
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# gets all letter rectangle coordinates from letter.png image and returns n first
def get_letter_rectangles(x, n):
    vertical_projection = x.mean(axis=0)
    horizontal_projection = x.mean(axis=1)

    xs = []

    for x_t in range(len(horizontal_projection) - 1):
        if horizontal_projection[x_t] - horizontal_projection[x_t + 1] != 0:
            if horizontal_projection[x_t] == 255:
                xs.append(x_t + 1)
            if horizontal_projection[x_t + 1] == 255:
                xs.append(x_t)

    ys = []

    for y_t in range(len(vertical_projection) - 1):
        if vertical_projection[y_t] - vertical_projection[y_t + 1] != 0:
            if vertical_projection[y_t] == 255:
                ys.append(y_t + 1)
            if vertical_projection[y_t + 1] == 255:
                ys.append(y_t)
    
    x_pairs = []
    for i in range(0, len(xs), 2):
        x_pairs.append([xs[i], xs[i + 1]]) 

    y_pairs = []
    for i in range(0, len(ys), 2):
        y_pairs.append([ys[i], ys[i + 1]]) 
    
    total_rects = len(x_pairs) * len(y_pairs)
    rects = np.zeros((total_rects, 4), dtype=np.int64)
    idx = 0

    for x_pair in x_pairs:
        for y_pair in y_pairs:
            rects[idx] = [int(x_pair[0]), int(x_pair[1]), int(y_pair[0]), int(y_pair[1])]
            idx += 1

    return rects[:n]

# gets a rectangle part of the image
def get_rectangle(x, rect):
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    image_part = x[x1:x2 + 1, y1:y2 + 1]

    return image_part

# discards white edges on an image
def crop_white_part(x):
    vertical_projection = x.mean(axis=0)
    horizontal_projection = x.mean(axis=1)

    x_start = 0
    x_end = x.shape[0] - 1
    y_start = 0
    y_end = x.shape[1] - 1
    for x_t in range(len(horizontal_projection) - 1):
        if horizontal_projection[x_t] - horizontal_projection[x_t + 1] != 0:
            if horizontal_projection[x_t] == 255:
                x_start = x_t + 1
            if horizontal_projection[x_t + 1] == 255:
                x_end = x_t
    for y_t in range(len(vertical_projection) - 1):
        if vertical_projection[y_t] - vertical_projection[y_t + 1] != 0:
            if vertical_projection[y_t] == 255:
                y_start = y_t + 1
            if vertical_projection[y_t + 1] == 255:
                y_end = y_t

    image_part = x[x_start:x_end + 1, y_start:y_end + 1]

    return image_part, (x_start, x_end, y_start, y_end)

def getcontour(x):
    # convert to binary image
    x_padded = x
    # double dimensions to eliminate gaps
    x_padded = cv2.resize(x_padded, (100, 100), interpolation=cv2.INTER_LINEAR)
    # pad each image 10 pixels each side for dilation etc
    x_padded = np.pad(x_padded, 10, "constant", constant_values=(255, 255))
    threshold = 140
    _, letter = cv2.threshold(x_padded, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # keep original
    letter_original = np.copy(letter)

    # # compute dilated version
    dilation_kernel = np.ones((3, 3), np.uint8)
    letter = cv2.bitwise_not(letter)
    letter = cv2.dilate(letter, dilation_kernel, iterations=1)
    letter = cv2.bitwise_not(letter)
    
    # subtract original version
    letter = letter - letter_original

    # apply thinning
    letter = letter * 255
    letter = cv2.ximgproc.thinning(letter)
    letter = cv2.bitwise_not(letter)

    # determine the n contours
    contour_points = np.transpose((letter == 0).nonzero())

    contours = []

    for point in contour_points:
        # check if it is in any existing contour
        added = False
        for i in range(len(contours)):
            if added:
                continue
            for current_point in contours[i]:
                if added:
                    continue
                if (abs(point[0] - current_point[0]) <= 1) and (abs(point[1] - current_point[1]) <= 1):
                    contours[i].append([point[0], point[1]])
                    added = True
        
        if not added:
            # else create a new one
            contours.append([[point[0], point[1]]])
    
    # merge contours 
    changed = True
    while changed:
        changed = False

        for i in range(len(contours)):
            if changed:
                continue
            for j in range(len(contours)):
                if changed:
                    continue
                for p1 in contours[i]:
                    if changed:
                        continue
                    for p2 in contours[j]:
                        if changed:
                            continue
                        if i != j and (abs(p1[0] - p2[0]) <= 1) and (abs(p1[1] - p2[1]) <= 1):
                            contours[i].extend(contours[j])
                            contours.pop(j)
                            changed = True
    
    def sorting_func(x1, x2):
        if x1[0] > x2[0]:
            return 1
        elif x1[0] < x2[0]:
            return -1
        else:
            if x1[1] > x2[1]:
                return 1
            elif x1[1] < x2[1]:
                return -1
            else: 
                return 0

    # sort first by x then by y
    for i in range(len(contours)):
        contours[i] = sorted(contours[i], key=cmp_to_key(sorting_func))

    return contours

def demo_contours():
    filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/letters.png"
    image_file = Image.open(filename)
    image_data = np.asarray(image_file)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    rects = get_letter_rectangles(image_data, 26)

    contour_colors = [
        [255, 0, 0], # red
        [0, 255, 0], # green
        [0, 0, 255], # blue
    ]

    # a, e, f, l
    requested_letters = [0, 4, 5, 11]

    dispTotal = []

    for rl in requested_letters:
        image_part = get_rectangle(image_data, rects[rl])
        letter, _ = crop_white_part(image_part)
        contours = getcontour(letter)

        disp = np.ones((120, 120, 3))

        for i in range(len(contours)):
            color = contour_colors[i]
            for point in contours[i]:
                disp[point[0]][point[1]] = color
        
        dispTotal.append(disp)
    
    plt.figure("Example letters: a, e, f, l")
    plt.subplot(221)
    plt.imshow(dispTotal[0])
    plt.subplot(222)
    plt.imshow(dispTotal[1])
    plt.subplot(223)
    plt.imshow(dispTotal[2])
    plt.subplot(224)
    plt.imshow(dispTotal[3])
    plt.savefig("letters_contours.png")
    plt.show()