import numpy as np
import cv2
import imutils
from imutils import face_utils
import dlib
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from matplotlib.path import Path
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
path = 'Dataset/Test/General'

eye_bags_red_mean = 0.1277055564508918
eye_bags_green_mean = 0.13208364114346535
eye_bags_green_dev = 0.0890832192370234
eye_bags_blue_mean = 0.14743503767390834
eye_bags_blue_dev = 0.08785469727806773

blue_lips_mean = 18.747098259964073
blue_lips_diff_mean = 3.382342778579465
blue_lips_diff_dev = 2.152412921588228

red_eyes_mean = 3.2140258841557534
red_eyes_dev = 3.143526693485413

asymmetric_eyes_mean = 0.6672306321257245
asymmetric_eyes_dev = 0.61221062671785326
asymmetric_mouth_mean = 2.0594594594594593
asymmetric_mouth_dev = 1.7893769804281592
asymmetric_eyebrows_mean = 2.4918918918918918
asymmetric_eyebrows_dev = 1.877660761228870

depressed_mouth_mean = 6.504545454545455
depressed_mouth_dev = 5.11725932264592

dry_lips_mean = 437.59194208515015
dry_lips_dev = 93.44890149593988
dry_lips_open_mouth_diff_mean = 178.85595374253367
dry_lips_open_mouth_diff_dev = 97.66810118612692
dry_lips_open_mean = 2.239130434782609
dry_lips_open_dev = 3.311277059624499


def get_rgb(photo):
    rgb_r = 0
    rgb_g = 0
    rgb_b = 0
    for i in range(photo.shape[0]):
        for j in range(photo.shape[1]):
            pixel = photo[i][j]
            a1, a2, a3 = pixel / 255
            rgb = sRGBColor(a1, a2, a3)
            rgb_r += rgb.rgb_r
            rgb_b += rgb.rgb_b
            rgb_g += rgb.rgb_g

    size = photo.shape[0] * photo.shape[1]

    return rgb_r / size, rgb_g / size, rgb_b / size


def calculate_a_sum(photo):
    a = []
    for i in range(photo.shape[0]):
        for j in range(photo.shape[1]):
            pixel = photo[i][j]
            a1, a2, a3 = pixel / 255
            rgb = sRGBColor(a1, a2, a3)
            if rgb.rgb_r != 0 or rgb.rgb_g != 0 or rgb.rgb_b != 0:
                a.append(rgb_to_cielab(pixel).lab_a)

    av_a = sum(a)
    return av_a, len(a)


def rgb_to_cielab(a):
    # a is a pixel with RGB coloring
    a1, a2, a3 = a / 255

    color1_rgb = sRGBColor(a1, a2, a3)

    color1_lab = convert_color(color1_rgb, LabColor)

    return color1_lab


def crop_image(part, link):
    vertices = part

    image = cv2.imread(link)
    img = imutils.resize(image, width=500)

    # from vertices to a matplotlib path
    path = Path(vertices)

    # create a mesh grid for the whole image, you could also limit the
    # grid to the extents above, I'm creating a full grid for the plot below
    x, y = np.mgrid[:img.shape[1], :img.shape[0]]
    # mesh grid to a list of points
    points = np.vstack((x.ravel(), y.ravel())).T

    # select points included in the path
    mask = path.contains_points(points)

    # reshape mask for display
    img_mask = mask.reshape(x.shape).T

    # masked image
    img *= img_mask[..., None]
    return img


def highest_point(arr):
    arr_y = []
    for point in arr:
        arr_y.append(point[1])
    return min(arr_y)


def calc_not_zero(edges):
    not_zero = 0
    for edge in edges:
        for item in edge:
            if item != 0:
                not_zero += 1
    return not_zero


def eye_bags_detection(left_eye, right_eye, image):
    (x, y, w, h) = cv2.boundingRect(np.array([left_eye]))
    coeff = 5
    roi = image[y + h + coeff:y + 2 * h + coeff, x:x + w]
    red_right_eye, green_right_eye, blue_right_eye = get_rgb(roi)
    roi_skin = image[y + 2 * h + coeff:y + 4 * h + coeff, x:x + w]
    red_skin_right_eye, green_skin_right_eye, blue_skin_right_eye = get_rgb(roi_skin)

    (x, y, w, h) = cv2.boundingRect(np.array([right_eye]))
    coeff = 5
    roi = image[y + h + coeff:y + 2 * h + coeff, x:x + w]
    red_left_eye, green_left_eye, blue_left_eye = get_rgb(roi)
    roi_skin = image[y + 2 * h + coeff:y + 4 * h + coeff, x:x + w]
    red_skin_left_eye, green_skin_left_eye, blue_skin_left_eye = get_rgb(roi_skin)

    red_diff_right = abs(red_right_eye - red_skin_right_eye)
    red_diff_left = abs(red_left_eye - red_skin_left_eye)
    green_diff_right = abs(green_right_eye - green_skin_right_eye)
    green_diff_left = abs(green_left_eye - green_skin_left_eye)
    blue_diff_right = abs(blue_right_eye - blue_skin_right_eye)
    blue_diff_left = abs(blue_left_eye - blue_skin_left_eye)

    red_diff = red_diff_right + red_diff_left
    green_diff = green_diff_right + green_diff_left
    blue_diff = blue_diff_right + blue_diff_left

    if red_diff > eye_bags_red_mean or (green_diff > eye_bags_green_mean + 0.089 * eye_bags_green_dev and blue_diff >
                                        eye_bags_blue_mean + 0.088 * eye_bags_blue_dev):
        return True
    return False


def blue_lips_detection(lips, image):
    img = crop_image(lips, image)
    a_lip, sum = calculate_a_sum(img)
    if a_lip / sum < blue_lips_mean - (blue_lips_diff_mean + blue_lips_diff_dev) * 0.85:
        return True
    return False


def red_eyes_detection(left_eye, right_eye, image):
    img = crop_image(left_eye, image)
    a_right_eye, sum_right_eye = calculate_a_sum(img)

    img = crop_image(right_eye, image)
    a_left_eye, sum_left_eye = calculate_a_sum(img)
    a_right = a_right_eye / sum_right_eye
    a_left = a_left_eye / sum_left_eye

    if a_right > red_eyes_mean + red_eyes_dev or a_left > red_eyes_mean + red_eyes_dev:
        return True
    return False


def asymmetric_face_detection(left_eye, right_eye, left_eyebrow, right_eyebrow, mouth_left, mouth_right):
    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")
    # compute the angle between the eye centroids
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle_eyes = np.degrees(np.arctan2(dY, dX)) - 180

    max_left_eyebrow = highest_point(left_eyebrow)
    max_right_eyebrow = highest_point(right_eyebrow)

    eyes = abs(180 - abs(angle_eyes))
    mouth = abs(mouth_left - mouth_right)
    eyebrows = abs(max_left_eyebrow - max_right_eyebrow)

    if eyes > asymmetric_eyes_mean + 2.14 * asymmetric_eyes_dev or mouth > asymmetric_mouth_mean + 2.14 * \
            asymmetric_mouth_dev or eyebrows > asymmetric_eyebrows_mean + 1.88 * asymmetric_eyebrows_dev:
        return True
    return False


def depression_detection(mouth, mouth_right_edge, mouth_left_edge):
    mouth_center = mouth.mean(axis=0).astype("int")
    mouthh = (mouth_center[1] - mouth_left_edge + mouth_center[1] - mouth_right_edge) / 2
    if mouthh < depressed_mouth_mean - 1.28 * depressed_mouth_dev:
        return True
    return False


def dry_lips_detection(mouth, image, mouth_up, mouth_down):
    img = crop_image(mouth, image)
    edges = cv2.Canny(img, 100, 200)
    mouth_diff = mouth_down - mouth_up
    lips = calc_not_zero(edges)

    if mouth_diff > dry_lips_open_mean + 2 * dry_lips_open_dev:
        lips -= dry_lips_open_mouth_diff_mean
        lips -= 0.5 * dry_lips_open_mouth_diff_dev
    if lips > dry_lips_mean + dry_lips_dev:
        return True
    return False


def detection(link):
    try:
        image = cv2.imread(link)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        print("Invalid file")

    else:
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        if len(rects) == 0:
            print("No face detected")
            return 0
        else:
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                if eye_bags_detection(shape[36:42], shape[42:48], image):
                    print("eye bags detected")
                if blue_lips_detection(shape[48:68], link):
                    print("blue lips detected")
                if red_eyes_detection(shape[36:42], shape[42:48], link):
                    print("red eyes detected")
                if asymmetric_face_detection(shape[36:48], shape[42:48], shape[18:22], shape[23:27], shape[48][1], shape[54][1]):
                    print("asymmetric face detected")
                if depression_detection(shape[48:68], shape[48][1], shape[54][1]):
                    print("depression detected")
                if dry_lips_detection(shape[48:68], link, shape[66][1], shape[62][1]):
                    print("dry lips detected")


for image_path in os.listdir(path):
    input_path = os.path.join(path, image_path)
    print(image_path)
    detection(input_path)