import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from preprocessing import findRotationAngle, rotateImage
from train_knn_model import remove_white_edges
from contours import get_rectangle, getcontour

def readtext(x, models, Ns):
    # first fix rotation of the image
    rot_angle = findRotationAngle(x)
    image = rotateImage(x, -rot_angle, "nearest")
    plt.imshow(image)
    plt.savefig("text_read_normal_rot.png")
    plt.show()
    print(f"Image for text reading was rotated {rot_angle * 180.0 / np.pi} degrees")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

    text = []

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
        # if line_idx != 16:
        #     continue
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
            word_txt = []
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
                # letter = cv2.resize(letter, (70, 70), interpolation=cv2.INTER_LINEAR)
                contours = getcontour(letter) 

                model = models[len(contours) - 1]
                N_t = Ns[len(contours) - 1]

                fft_contours = []
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
                
                final_contours = np.array(fft_contours).flatten()

                prediction = model.predict([final_contours])
                
                word_txt.append(prediction.item())
                
            text_line.append("".join(word_txt)) 

        text.append(" ".join(text_line))

    text = "\n".join(text)

    return text