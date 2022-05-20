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
    img = normalize(img)
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
    for i, contour in enumerate(contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        ar = w / float(h)
        if len(approx) == 4 and area > 1000 and (ar > 0.85 and ar < 1.3):
            ROI = original[y : y + h, x : x + w]
            # cv2.imwrite(f"ROI-{i}.png", ROI)
            candidates.append((Point(x, y), Point(x + w, y + h)))
    return candidates


def c_mid(cand):
    p1, p2 = cand
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2


def c_extreme(candidates, direction):
    """Returns extremal candidate in direction of vector [a, b]"""
    i = np.argmax([np.dot(c_mid(c), direction) for c in candidates])
    return candidates[i]


def visualize_candidates(candidates, image):
    green = (36, 255, 12)
    for p1, p2 in candidates:
        cv2.rectangle(image, (p1.x, p1.y), (p2.x, p2.y), green, 3)

    cv2.imshow("Candidates", image)
    cv2.waitKey()


def normalize(image):
    original = image.copy()
    thresh = binarize(image)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = find_candidates(original, image, close)
    visualize_candidates(candidates, image)

    # select extremal candidate boxes
    p_tl = c_mid(c_extreme(candidates, [-1, -1]))  # top-left
    p_tr = c_mid(c_extreme(candidates, [+1, -1]))
    p_bl = c_mid(c_extreme(candidates, [-1, +1]))
    p_br = c_mid(c_extreme(candidates, [+1, +1]))  # bottom-right

    # Apply projective transformation mapping extermeal points to corners
    # https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
    rows, cols, _ = image.shape
    pts1 = np.float32([p_tl, p_tr, p_bl, p_br])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (cols, rows))

    cv2.imshow("Normalized", image)
    cv2.waitKey()


def save_np_array(img, path):
    im = Image.fromarray(img)
    im.save(path)


if __name__ == "__main__":
    cli()
