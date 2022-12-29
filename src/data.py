import cv2
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# custom imports
from config import *


def load_image_list(img_files, gray=False):
    """
    Diese Funktion lädt eine Liste von Bildern und gibt diese als Liste zurück.
    :param img_files --> die Liste der Bilder, die geladen werden sollen
    :param gray --> gibt an, ob die Bilder in Graustufen geladen werden sollen
    """
    imgs = []
    if gray:
        for image_file in img_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            imgs += [img]

    else:
        for image_file in img_files:
            imgs += [cv2.imread(image_file)]
    return imgs


def clahe_images(img_list):
    """
    diese Funktion ist die clahe images Funktion, die einen clahe threshold auf die Eingabebilder anwendet.
    :param img_files --> Liste der Bilder, die verarbeitet werden sollen

    :return img_list --> die verarbeiteten Bilder
    """
    for i, img in enumerate(img_list):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img_list[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_list


def preprocess_image(imgs, padding=padding[1]):
    """
    Diese Funktion fügt einen Padding zu den Eingabebildern hinzu.
    Dazu zählen die Bilder, die Masken und die Edges.
    :param imgs --> die Liste der Bilder, die verarbeitet werden sollen
    :param padding --> das Padding, das hinzugefügt werden soll
    :return imgs --> die verarbeiteten Bilder
    """
    imgs = [
        np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        for img in imgs
    ]
    return imgs


def preprocess_data(imgs, mask, edge=None, padding=padding[1]):
    """
    Diese Funktion fügt einen Padding zu den Eingabebildern hinzu.
    Dazu zählen die Bilder, die Masken und die Edges.
    Sie ist wie die Funktion preprocess_image
    :param imgs --> die Liste der Bilder, die verarbeitet werden sollen
    :param mask --> die Liste der Masken, die verarbeitet werden sollen
    :param edge --> die Liste der Edges, die verarbeitet werden sollen
    :param padding --> das Padding, das hinzugefügt werden soll
    :return imgs --> die verarbeiteten Bilder
    """

    imgs = [
        np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        for img in imgs
    ]
    mask = [
        np.pad(mask, ((padding, padding), (padding, padding)), mode="constant")
        for mask in mask
    ]
    if edge is not None:
        edge = [
            np.pad(edge, ((padding, padding), (padding, padding)), mode="constant")
            for edge in edge
        ]

    if edge is not None:
        return imgs, mask, edge

    return imgs, mask


def load_data(img_list, mask_list, edge_list=None, padding=padding[1]):
    """
    Das Laden und Preprocessen der Daten wird in dieser Funktion durchgeführt.

    :param img_list --> die Liste der Bilder, die geladen werden sollen
    :param mask_list --> die Liste der Masken, die geladen werden sollen
    :param edge_list --> die Liste der Edges, die geladen werden sollen
    :param padding --> das Padding, das hinzugefügt werden soll

    :return imgs(imgs, masken und edges [Falls vorhanden]) --> die verarbeiteten Bilder
    """
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)

    mask = load_image_list(mask_list, gray=True)
    if edge_list:
        edge = load_image_list(edge_list, gray=True)
    else:
        edge = None

    return preprocess_data(imgs, mask, edge, padding=padding)


def load_image(img_list, padding=padding[1]):
    """
    Das Laden und Preprocessen der Bilder wird in dieser Funktion durchgeführt.

    :param img_list --> die Liste der Bilder, die geladen werden sollen
    :param padding --> das Padding, das hinzugefügt werden soll

    :return imgs --> die verarbeiteten Bilder
    """
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)
    return preprocess_image(imgs, padding=padding)


def aug_lum(image, factor=None):
    """
    Die Helligkeit Augmentierung wird in dieser Funktion durchgeführt.
    Die Lumination wird mit einem Faktor multipliziert.
    :param image --> das Bild, das verarbeitet werden soll
    :param factor --> der Faktor, mit dem die Helligkeit verändert werden soll (default: 0.5 * zufallige Zahl)
    :Return image --> das verarbeitete Bild
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    if factor is None:
        lum_offset = 0.5 + np.random.uniform()
    else:
        lum_offset = factor

    hsv[..., 2] = hsv[..., 2] * lum_offset
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aug_img(image):
    """
    Farbaugmentierung wird in dieser Funktion durchgeführt.
    :param image --> das Bild, das verarbeitet werden soll
    :Return image --> das verarbeitete Bild
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    hue_offset = 0.8 + 0.4 * np.random.uniform()
    sat_offset = 0.5 + np.random.uniform()
    lum_offset = 0.5 + np.random.uniform()

    hsv[..., 0] = hsv[..., 0] * hue_offset
    hsv[..., 1] = hsv[..., 1] * sat_offset
    hsv[..., 2] = hsv[..., 2] * lum_offset

    hsv[..., 0][hsv[..., 0] > 255] = 255
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 2][hsv[..., 2] > 255] = 255

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def train_generator(
    imgs,
    mask,
    edge=None,
    scale_range=None,
    padding=padding[1],
    input_size=input_shape[0],
    output_size=output_shape[0],
    skip_empty=False,
):
    """
    Diese Funktion generiert die Trainingsdaten.

    :param imgs --> die Liste der Bilder, die verarbeitet werden sollen
    :param mask --> die Liste der Masken, die verarbeitet werden sollen
    :param edge --> die Liste der Edges, die verarbeitet werden sollen
    :param scale_range --> der Bereich, in dem die Bilder skaliert werden sollen (i, j)
    :param padding --> das Padding, das hinzugefügt werden soll
    :param input_size --> die Größe des Inputs
    :param output_size --> die Größe des Outputs
    :param skip_empty --> ob leere Masken übersprungen werden sollen

    :return img --> yields das Bild, die Maske und die Edge chips jedes Mal, wenn die Funktion aufgerufen wird
    """
    if scale_range is not None:
        scale_range = [1 - scale_range, 1 + scale_range]
    while True:
        # select which type of cell to return
        chip_type = np.random.choice([True, False])

        while True:
            # pick random image
            i = np.random.randint(len(imgs))

            # pick random central location in the image (200 + 196/2)
            center_offset = padding + (output_size / 2)
            x = np.random.randint(center_offset, imgs[i].shape[0] - center_offset)
            y = np.random.randint(center_offset, imgs[i].shape[1] - center_offset)

            # scale the box randomly from x0.8 - 1.2x original size
            scale = 1
            if scale_range is not None:
                scale = scale_range[0] + (
                    (scale_range[0] - scale_range[0]) * np.random.random()
                )

            # find the edges of a box around the image chip and the mask chip
            chip_x_l = int(x - ((input_size / 2) * scale))
            chip_x_r = int(x + ((input_size / 2) * scale))
            chip_y_l = int(y - ((input_size / 2) * scale))
            chip_y_r = int(y + ((input_size / 2) * scale))

            mask_x_l = int(x - ((output_size / 2) * scale))
            mask_x_r = int(x + ((output_size / 2) * scale))
            mask_y_l = int(y - ((output_size / 2) * scale))
            mask_y_r = int(y + ((output_size / 2) * scale))

            # take a slice of the image and mask accordingly
            temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
            temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
            if edge is not None:
                temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

            if skip_empty:
                if ((temp_mask > 0).sum() > 5) is chip_type:
                    continue

            # resize the image chip back to 380 and the mask chip to 196
            temp_chip = cv2.resize(
                temp_chip, (input_size, input_size), interpolation=cv2.INTER_CUBIC
            )
            temp_mask = cv2.resize(
                temp_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST
            )
            if edge is not None:
                temp_edge = cv2.resize(
                    temp_edge,
                    (output_size, output_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            # randomly rotate (like below)
            rot = np.random.randint(4)
            temp_chip = np.rot90(temp_chip, k=rot, axes=(0, 1))
            temp_mask = np.rot90(temp_mask, k=rot, axes=(0, 1))
            if edge is not None:
                temp_edge = np.rot90(temp_edge, k=rot, axes=(0, 1))

            # randomly flip
            if np.random.random() > 0.5:
                temp_chip = np.flip(temp_chip, axis=1)
                temp_mask = np.flip(temp_mask, axis=1)
                if edge is not None:
                    temp_edge = np.flip(temp_edge, axis=1)

            # randomly luminosity augment
            temp_chip = aug_lum(temp_chip)

            # randomly augment chip
            temp_chip = aug_img(temp_chip)

            # rescale the image
            temp_chip = temp_chip.astype(np.float32) * 2
            temp_chip /= 255
            temp_chip -= 1

            # later on ... randomly adjust colours
            if edge is not None:
                yield temp_chip, (
                    (temp_mask > 0).astype(float)[..., np.newaxis],
                    (temp_edge > 0).astype(float)[..., np.newaxis],
                )
            else:
                yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis])
            break


def test_chips(
    imgs,
    mask,
    edge=None,
    padding=padding[1],
    input_size=input_shape[0],
    output_size=output_shape[0],
):
    """
    Diese Funktion generiert die Testdaten.
    :param imgs --> die Eingabebilder
    :param mask --> die Eingabemasken
    :param edge --> die Eingabeedges, falls vorhanden (nur rote Blutkörperchen)
    :param padding --> das Padding, das auf jedes Bild angewendet wird
    :param input_size --> die Eingabegröße
    :param output_size --> die Ausgabegröße

    :return chips --> yieldt ein Bildchip, Maskenchip und Edgechip, wenn es aufgerufen wird
    """
    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(
            center_offset, imgs[i].shape[0] - input_size / 2, output_size
        ):
            for y in np.arange(
                center_offset, imgs[i].shape[1] - input_size / 2, output_size
            ):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                if edge is not None:
                    yield temp_chip, (
                        (temp_mask > 0).astype(float)[..., np.newaxis],
                        (temp_edge > 0).astype(float)[..., np.newaxis],
                    )
                else:
                    yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis])
                break


def slice_image(
    imgs, padding=padding[1], input_size=input_shape[0], output_size=output_shape[0]
):
    """
    Diese Funktion schneidet jedes Bild in Bildchips.
    :param imgs --> die Eingabebilder
    :param padding --> das Padding, das auf jedes Bild angewendet wird
    :param input_size --> die Eingabegröße
    :param output_size --> die Ausgabegröße

    :return list tuple (list, list, list) --> Die tuple liste der Ausgabe (Bildchips, Maskenchips und Edgechips)
    """
    img_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(
            center_offset, imgs[i].shape[0] - input_size / 2, output_size
        ):
            for y in np.arange(
                center_offset, imgs[i].shape[1] - input_size / 2, output_size
            ):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
    return np.array(img_chips)


def slice(
    imgs,
    mask,
    edge=None,
    padding=padding[1],
    input_size=input_shape[0],
    output_size=output_shape[0],
):
    """
    Wie oben nur für mehrere Bilder.
    """
    img_chips = []
    mask_chips = []
    if edge is not None:
        edge_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(
            center_offset, imgs[i].shape[0] - input_size / 2, output_size
        ):
            for y in np.arange(
                center_offset, imgs[i].shape[1] - input_size / 2, output_size
            ):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
                mask_chips += [(temp_mask > 0).astype(float)[..., np.newaxis]]
                if edge is not None:
                    edge_chips += [(temp_edge > 0).astype(float)[..., np.newaxis]]

    img_chips = np.array(img_chips)
    mask_chips = np.array(mask_chips)
    if edge is not None:
        edge_chips = np.array(edge_chips)

    if edge is not None:
        return img_chips, mask_chips, edge_chips

    return img_chips, mask_chips


def generator(img_list, mask_list, edge_list=None, type="train"):
    """
    Diese Funktion ist der Generator, der die Liste der Bilder, Masken und Kanten zu dem train_generator und test_chips Funktionen liefert.
    :param img_list --> die Liste der Bilder
    :param mask_list --> die Liste der Masken
    :param edge_list --> die Liste der Kanten, falls vorhanden
    :param type --> kann entweder train oder test sein, wird verwendet um zu bestimmen welche Generator Funktion aufgerufen wird

    :return tensorflow dataset --> die generierten Funktionen, die an tensorflow übergeben werden
    """
    if cell_type == "rbc":
        img, mask, edge = load_data(img_list, mask_list, edge_list)
    elif cell_type == "wbc" or cell_type == "plt":
        img, mask = load_data(img_list, mask_list)
        edge = None

    def gen():
        if type == "train":
            return train_generator(
                img,
                mask,
                edge,
                padding=padding[0],
                input_size=input_shape[0],
                output_size=output_shape[0],
            )
        elif type == "test":
            return test_chips(
                img,
                mask,
                edge,
                padding=padding[0],
                input_size=input_shape[0],
                output_size=output_shape[0],
            )

    # load train dataset to tensorflow for training
    if cell_type == "rbc":
        return tf.data.Dataset.from_generator(
            gen,
            (tf.float64, ((tf.float64), (tf.float64))),
            (input_shape, (output_shape, output_shape)),
        )
    elif cell_type == "wbc" or cell_type == "plt":
        return tf.data.Dataset.from_generator(
            gen, (tf.float64, (tf.float64)), (input_shape, (output_shape))
        )
