"""Preprocess a photo"""

from typing import List, NamedTuple, Tuple

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class Point(NamedTuple):
    x: float
    y: float


@click.command()
@click.argument("filename")
def cli(filename: str):
    img = cv2.imread(filename)
    bb = find_corners(img)
    # img = crop_image(img, bb[0], bb[1])
    save_np_array(img, "detected-corners.jpeg")


def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def find_candidates(original, image, close) -> List[Tuple[Point, Point]]:
    candidates = []
    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        ar = w / float(h)
        if len(approx) == 4 and area > 1000 and (ar > 0.85 and ar < 1.3):

            ROI = original[y : y + h, x : x + w]
            cv2.imwrite("ROI.png", ROI)
            candidates.append((Point(x, y), Point(x + w, y + h)))
    return candidates


def visualize_candidates(candidates, image):
    green = (36, 255, 12)
    for p1, p2 in candidates:
        cv2.rectangle(image, (p1.x, p1.y), (p2.x, p2.y), green, 3)
    cv2.imshow("image", image)
    cv2.waitKey()


def find_corners(image) -> Tuple[Point, Point]:
    original = image.copy()
    thresh = binarize(image)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = find_candidates(original, image, close)
    visualize_candidates(candidates, image)

    # hull
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = 0, 0
    for p1, p2 in candidates:
        min_x = min(min_x, p1.x)
        min_y = min(min_y, p1.y)
        max_x = max(max_x, p2.x)
        max_y = max(max_y, p2.y)

    return Point(min_x, min_y), Point(max_x, max_y)


def crop_image(image, p1: Point, p2: Point):
    cropped = image[p1.x : p2.x, p1.y : p2.y]
    return cropped


def save_np_array(img, path):
    im = Image.fromarray(img)
    im.save(path)


if __name__ == "__main__":
    cli()
