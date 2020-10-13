#!/usr/bin/env python3

import argparse
from collections import namedtuple
import os

# Disabling tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import cv2
import tensorflow as tf

size = namedtuple("size", "w h")
rectangle = namedtuple("rectangle", "x y w h")


def get_verbose_imwriter():
    step = 0

    def verbose_imwriter(img, step_info=None):
        nonlocal step
        global VERBOSE
        if VERBOSE:
            if step_info is None:
                cv2.imwrite('./step{}.png'.format(step), img)
            else:
                cv2.imwrite('./step{}-{}.png'.format(step, step_info), img)
            step += 1

    return verbose_imwriter


def rectangle_area(r):
    return r.w * r.h


PLATE_LENGTH = 7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License Plate Detection and Recognition')
    parser.add_argument('file', type=str, help='image file')
    parser.add_argument('-s', '--approx_size', nargs=2, type=int,
                        dest="s", help='approximate size of plate(the default value is (w / 3, h / 19))')
    parser.add_argument('-L', '--plate-length', type=int,
                        dest="l", help='an integer for the accumulator(the default value of plate_lengthi is 7)')
    parser.add_argument("-v", "--verbose", dest="verbosity",
                        action='store_true', help="save steps' images in current directory")
    args = parser.parse_args()

    file = args.file
    VERBOSE = args.verbosity
    img = cv2.imread(file)
    w, h, _ = img.shape
    img_size = size(w, h)
    approx_size = size(w / 3, h / 19)
    if args.s is not None:
        approx_size = size(args.s[0], args.s[1])
    if args.l is not None:
        PLATE_LENGTH = args.l

    viw = get_verbose_imwriter()
    viw(img, "original")

    # We don't need colors
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    viw(img_gray, "grayscale")
    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    img_bilateral = cv2.bilateralFilter(img_gray, 10, 300, 300)
    viw(img_bilateral, "bilateral")
    # Find edges of the grayscale image
    edged = cv2.Canny(img_bilateral, 170, 200)
    viw(edged, "edged")

    # Find contours based on Edges
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_list = []
    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        similarity = (approx_size.w - w) ** 2 + (approx_size.h - h) ** 2
        contours_list.append((similarity, (x, y, w, h)))
    x, y, w, h = sorted(contours_list, key=lambda x: x[0])[0][1]
    plate_rect = rectangle(x, y, w, h)
    img_plate = img[y:y + h, x:x + w]
    viw(img_plate, "img_plate")

    # We don't need colors again
    plate_gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
    viw(plate_gray, "gray_scale_plate")
    # binarize
    ret, thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY_INV)
    viw(thresh, "binarize_plate")
    # find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding box
    rectangles = [rectangle(*cv2.boundingRect(contour)) for contour in contours]
    # filter for rectangles that their height is greater than their width
    rectangles = filter(lambda r: r.h > r.w, rectangles)
    # sort by area and pick 8
    rectangles = sorted(rectangles, key=rectangle_area, reverse=True)[0:PLATE_LENGTH]
    # sort by width
    rectangles = sorted(rectangles, key=lambda r: r.w)
    # sort by x
    rectangles = sorted(rectangles, key=lambda r: r.x)

    img_chars = []
    for i, rect in enumerate(rectangles):
        x, y, w, h = rect
        # Getting ROI
        roi = thresh[y:y + h, x:x + w]
        # show ROI
        img_chars.append(roi)
        viw(roi, "roi_img{}".format(i))
        cv2.rectangle(img_plate, (x, y), (x + w, y + h), (90, 0, 255), 2)

    viw(img, "end_of_detection")

    model = tf.keras.models.load_model("./ocr_model.h5", compile=False)

    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    np_string = []
    for img_char in img_chars:
        img = cv2.resize(img_char, (28, 28))
        img = img.astype("float32")
        img /= 255.0
        prediction = model.predict(cv2.resize(img, (28, 28)).reshape((1, 28, 28, 1)))
        i = prediction.argmax()
        np_string.append(chars[i])
    print("number plate = {}".format("".join(np_string)))
