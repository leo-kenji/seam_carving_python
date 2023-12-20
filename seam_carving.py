from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy import ndimage


@njit
def gray2rgb(img):
    # if np.max(img) > 1:
    img = np.copy(img).astype(np.float32)
    img = img / np.max(img)
    # (img,)*3 doesn't work in numba
    x = np.stack((img, img, img), axis=-1)
    return x


@njit
def rgb2gray(img):
    size = img.shape[0:2]
    new_img = np.zeros(size)
    for i, row in enumerate(new_img):
        for j, _ in enumerate(row):
            row[j] = np.mean(img[i, j])
    return new_img


def compute_energy(img):
    img = rgb2gray(img)
    sobel_h = ndimage.sobel(img, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(img, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    return magnitude


@njit
def compute_minimum_energy_map(energy):
    min_energy = np.zeros_like(energy)

    min_energy[0, :] = np.copy(energy[0])
    for i, row in enumerate(energy[1:], 1):
        for j, _ in enumerate(row):
            min_j = max(j - 1, 0)
            max_j = min(j + 2, len(row))
            min_energy[i, j] = energy[i, j] + np.min(min_energy[i - 1, min_j:max_j])
    return min_energy


@njit
def min_index_in_range(array, min_idx, max_idx):
    idx = min_idx
    min_value = array[min_idx]
    for i in range(min_idx, max_idx):
        if array[i] < min_value:
            min_value = array[i]
            idx = i

    return idx


@njit
def find_min_energy_path(energy):
    n_rows = energy.shape[0]
    idx = np.zeros(n_rows, dtype=np.int32)
    flipped = energy[::-1]
    idx[0] = np.argmin(flipped[0])
    for i, row in enumerate(flipped[1:], 1):
        last_j = idx[i - 1]
        min_j = max(last_j - 1, 0)
        max_j = min(last_j + 2, len(row))
        idx[i] = min_index_in_range(row, min_j, max_j)
    return idx[::-1]


@njit
def remove_path(img, a):
    new_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]))
    for i, row in enumerate(new_img):
        idx_not_included = a[i]
        row[:] = np.concatenate(
            (img[i, 0:idx_not_included], img[i, idx_not_included + 1 :]), axis=0
        )
    return new_img


def remove_columns(img, n_columns):
    img = np.copy(img)
    if img.dtype == np.uint8:
        img = img / 255
    for _ in range(n_columns):
        energy = compute_energy(img)
        min_energy = compute_minimum_energy_map(energy)
        path_to_be_removed = find_min_energy_path(min_energy)
        img = remove_path(img, path_to_be_removed)

    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    return img


def main() -> None:
    image_path = Path("Castle5565.jpg")
    img = iio.imread(image_path)
    img = img / 255

    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    axs[0].imshow(img)
    cropped = remove_columns(img, 1)
    axs[1].imshow(cropped)
    plt.show()


if __name__ == "__main__":
    main()
