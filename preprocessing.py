import numpy as np
import math
import cv2

# finds the largest change on the line projection
def calculate_spectrum_vertical_sum(x):
    # calculate spectrum
    img_fft = np.fft.fft2(x)
    img_fft = np.fft.fftshift(img_fft)
    img_fft_mag = np.abs(img_fft)
    c = 20
    spectrum = c * np.log(img_fft_mag)
    M = x.shape[0]
    N = x.shape[1]
    value_sum = np.mean(spectrum[0:M // 2, N // 2 - 5: N // 2 + 5])
    
    return value_sum

# kernel for bicubic interpolation
def bicubic_kernel(x):
    if abs(x) <= 1:
        return (1.5 * abs(x) * abs(x) * abs(x) - 2.5 * abs(x) * abs(x) + 1)
    elif 1 < abs(x) <= 2:
        return (-0.5 * abs(x) * abs(x) * abs(x) + 2.5 * abs(x) * abs(x) - 4 * abs(x) + 2)
    else:
        return 0

def findRotationAngle(x):
    angle = 0

    M = x.shape[0]
    N = x.shape[1]

    # first convert image to monochrome(we don't really care about coloring)
    monochrome_image = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) 
    # blur it
    kernel_size = 11
    monochrome_image_blurred = cv2.GaussianBlur(monochrome_image, (kernel_size, kernel_size), 0)

    # apply threshold
    threshold = 220
    _, monochrome_image_blurred = cv2.threshold(monochrome_image_blurred, threshold, 255, cv2.THRESH_BINARY)

    # calculating fft
    img_fft = np.fft.fft2(monochrome_image_blurred)
    # have it in the proper space 
    img_fft = np.fft.fftshift(img_fft)
    img_fft_mag = np.abs(img_fft)

    # calculate spectrum
    c = 20
    spectrum = c * np.log(1 + img_fft_mag)
    # remove DC term
    band = 20
    spectrum[M // 2 - band: M // 2 + band + 1, N // 2 - band: N // 2 + band + 1] = 0

    # find brightest pixel - estimate
    max_index = np.unravel_index(np.argmax(spectrum, axis=None), spectrum.shape)
    angle = np.pi / 2 - np.arctan2(- (max_index[0] - M // 2), max_index[1] - N // 2)

    # rough estimate for angle, search +- 10 degrees
    estimate_displacement = 5 * np.pi / 180.0
    search_range = [angle - estimate_displacement, angle + estimate_displacement]
    # resolution for search
    # can lower resolution (==make step higher) for faster speeds
    resolution = 0.5
    search_step = resolution * np.pi / 180.0

    # projection based here
    best_angle = angle
    current_top_vertical_spectrum_sum = calculate_spectrum_vertical_sum(monochrome_image_blurred)

    angles_to_search = np.arange(search_range[0], search_range[1], search_step)
        
    for angle_t in angles_to_search:
        # pass everything, calculate vertical projection
        rotated_image = rotateImage(x, -angle_t, "nearest")
        monochrome_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY) 
        kernel_size = 11
        monochrome_image_blurred = cv2.GaussianBlur(monochrome_image, (kernel_size, kernel_size), 0)
        threshold = 220
        _, monochrome_image_blurred = cv2.threshold(monochrome_image_blurred, threshold, 255, cv2.THRESH_BINARY)

        top_vertical_spectrum_sum = calculate_spectrum_vertical_sum(monochrome_image_blurred)
        
        # if its better, keep it
        if top_vertical_spectrum_sum >= current_top_vertical_spectrum_sum:
            current_top_vertical_spectrum_sum = top_vertical_spectrum_sum
            best_angle = angle_t

    # return angle
    return best_angle

# function to rotate the image from image center
# angle in radians
# use bicubic interpolation to produce good results
# or nearest neighboor to produce fast results
def rotateImage(x, angle, method="nearest"):
    # rotation matrix
    cosa = math.cos(angle)
    sina = math.sin(angle)
    # since its a small matrix, will not use rotation matrix but the formulas directly
    # rot = np.array([[cosa, sina], [-sina, cosa]]);

    M = x.shape[0]
    N = x.shape[1]

    center_x = M // 2
    center_y = N // 2

    # calculate 4 corners to determine new dimensions
    x1 = (0 - center_x) * cosa + (0 - center_y) * sina + center_x
    y1 = - (0 - center_x) * sina + (0 - center_y) * cosa + center_y
    x2 = (0 - center_x) * cosa + (N - center_y) * sina + center_x
    y2 = - (0 - center_x) * sina + (N - center_y) * cosa + center_y
    x3 = (M - center_x) * cosa + (0 - center_y) * sina + center_x
    y3 = - (M - center_x) * sina + (0 - center_y) * cosa + center_y
    x4 = (M - center_x) * cosa + (N - center_y) * sina + center_x
    y4 = - (M - center_x) * sina + (N - center_y) * cosa + center_y
    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.floor(x2)
    y2 = math.floor(y2)
    x3 = math.floor(x3)
    y3 = math.floor(y3)
    x4 = math.floor(x4)
    y4 = math.floor(y4)

    minx = min([x1, x2, x3, x4])
    maxx = max([x1, x2, x3, x4])
    miny = min([y1, y2, y3, y4])
    maxy = max([y1, y2, y3, y4])

    Mnew = maxx - minx
    Nnew = maxy - miny

    # assume that empty == white
    if len(x.shape) > 2:
        channels = x.shape[2]
        y = 255 * np.ones([Mnew, Nnew, channels])
    else:
        y = 255 * np.ones([Mnew, Nnew])

    for x_n in range(Mnew):
        for y_n in range(Nnew):
            # calculate original coordinates by rotating by negative angle
            # cos(-a) = cos(a) 
            # sin(-a) = -sin(a)
            x_p = (x_n - Mnew // 2) * cosa - (y_n - Nnew // 2) * sina + M// 2
            y_p = (x_n - Mnew // 2) * sina + (y_n - Nnew // 2) * cosa + N// 2

            # if they are valid coordinates (aka it is in the old image)
            # then process the image
            if x_p >= 0 and x_p < M and y_p >= 0 and y_p < N:
                # calculate pixel coordinates (integer)
                x_p_floored = math.floor(x_p)
                y_p_floored = math.floor(y_p)
                
                if method == "nearest":
                    y[x_n][y_n] = x[x_p_floored][y_p_floored]

                elif method == "bicubic":
                    # calculate the distance from the actual pixel
                    dx = x_p - x_p_floored
                    dy = y_p - y_p_floored

                    # find weight matrices
                    weights_x = np.array([bicubic_kernel(1 + dx), bicubic_kernel(dx), bicubic_kernel(1 - dx), bicubic_kernel(2 - dx)])
                    weights_y = np.array([bicubic_kernel(1 + dy), bicubic_kernel(dy), bicubic_kernel(1 - dy), bicubic_kernel(2 - dy)])

                    # get the pixels from the mask
                    channels = 1
                    if len(x.shape) > 2:
                        channels = x.shape[2]
                    pixels = np.zeros((4, 4, channels))
                    for i in range(-1, 3):
                        for j in range(-1, 3):
                            pixels[i + 1][j + 1] = x[max(0, min(x_p_floored + i, M - 1))][max(0, min(y_p_floored + j, N - 1))]
                    
                    # multiply them point to point with the precalculated weights
                    weighted_pixels = np.zeros_like(pixels)
                    for i in range(4):
                        for j in range(4):
                            weighted_pixels[i, j] = pixels[i, j] * weights_x[i] * weights_y[j]

                    # take their sum on x y axises
                    interpolated_pixel = np.sum(weighted_pixels, axis=(0, 1))
                    # set pixel on new rotated image
                    interpolated_pixel = np.maximum(0, np.minimum(255, interpolated_pixel))
                    y[x_n][y_n] = interpolated_pixel

    # return image
    y = y.astype(np.uint8)
    return y