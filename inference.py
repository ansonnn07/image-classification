# python inference_test.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import random
import os
import math
import time
import seaborn as sns
from utils import config
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_results(image_paths, actual_labels, predicted_labels, size=112):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    for idx in np.arange(len(image_paths)):
        ax = fig.add_subplot(N_ROWS, N_COLS, idx + 1, xticks=[], yticks=[])
        image = load_img(image_paths[idx]).resize((size, size))
        ax.imshow(image)
        fontColor = "black" if actual_labels[idx] == predicted_labels[idx] else "red"
        filename = image_paths[idx].split(os.path.sep)[-1]
        ax.set_title(
            " ".join(
                (
                    "Predicted:",
                    predicted_labels[idx],
                    "\nActual:",
                    actual_labels[idx],
                    f"\n{filename}",
                )
            ),
            fontdict={"size": 15, "color": fontColor},
        )
    plt.tight_layout(pad=0.8)
    plt.savefig(os.path.join(config.OUTPUTS_PATH, "sample_predictions.png"))
    plt.show()


def plot_top_losses(pred_probas, actual_labels, predicted_labels, size=112, top_k=5):
    print(f"[INFO] Plotting top {top_k} wrong predictions...")

    # Get top losses based on prediction probabilities
    wrong_pred_idxs = np.where(pred_labels != labels)[0]
    prob_idx_dict = {
        k: v for k, v in zip(wrong_pred_idxs, pred_probas[wrong_pred_idxs])
    }
    sorted_wrong_pred = sorted(
        prob_idx_dict.items(), key=lambda x: x[1].max(), reverse=True
    )
    print(sorted_wrong_pred[:top_k])

    # rows and cols to plot any number of images
    n_rows = int(top_k ** 0.5)
    n_cols = math.ceil(top_k / n_rows)

    figsize = get_figsize(n_rows, n_cols)
    fig = plt.figure(figsize=figsize, facecolor="white")
    for i, (img_idx, _) in enumerate(sorted_wrong_pred[:top_k]):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        image = load_img(imagePaths[img_idx]).resize((size, size))
        ax.imshow(image)
        fontColor = (
            "black" if actual_labels[img_idx] == predicted_labels[img_idx] else "red"
        )
        filename = imagePaths[img_idx].split(os.path.sep)[-1]
        ax.set_title(
            " ".join(
                (
                    "Predicted:",
                    predicted_labels[img_idx],
                    "\nActual:",
                    actual_labels[img_idx],
                    f"\n{filename}",
                )
            ),
            fontdict={"size": 15, "color": fontColor},
        )
    plt.tight_layout(pad=0.8)
    plt.savefig(os.path.join(config.OUTPUTS_PATH, "sample_predictions.png"))
    plt.show()


def get_figsize(nrows, ncols):
    return (3.8 * ncols, 5 * nrows)


# whether to plot the results
PLOT_IMAGES = True
# whether to run full evaluation on test set
FULL_EVAL = True
# whether to use ImageDataGenerator
USE_GEN = False

# flags to make sure things work properly
PLOT_IMAGES = False if FULL_EVAL else PLOT_IMAGES
USE_GEN = False if not FULL_EVAL else USE_GEN

# Plotting config

# Number of images to sample from each category (Standing VS NotStanding)
# Set it to a multiple of N_COLS to fill up the entire composite image
SAMPLE_SIZE = 8
# number of columns each of the image in the image montage
N_COLS = 4
# number of rows
N_ROWS = math.ceil(2 * SAMPLE_SIZE / N_COLS)
# Figure size, adjust this to fit images properly
FIGSIZE = get_figsize(N_ROWS, N_COLS)

# Load the trained model
# model = load_model(config.MODEL_PATH)


# Using testing dataset
basePath = config.TEST_PATH
standing_paths = os.path.join(basePath, "Standing")
not_standing_paths = os.path.join(basePath, "NotStanding")
if FULL_EVAL:
    imagePaths = list(paths.list_images(basePath))
else:
    imagePaths = random.sample(
        list(paths.list_images(standing_paths)), SAMPLE_SIZE
    ) + random.sample(list(paths.list_images(not_standing_paths)), SAMPLE_SIZE)


# initializing data for storing image data
# and labels for images
data = []
labels = []
n_images = len(imagePaths)
print(f"[INFO] Predicting for {n_images} images...")
start_time = time.time()

if USE_GEN:
    print("[INFO] Using ImageDataGenerator with flow_from_directory method...")
    valAug = ImageDataGenerator()
    # NOTE: this mean subtraction seems like does not do anything here
    # mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    # valAug.mean = mean
    testGen = valAug.flow_from_directory(
        config.TEST_PATH,
        class_mode="categorical",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=False,
        batch_size=config.BS,
    )
    testGen.reset()
    pred_probas = model.predict(testGen, steps=(n_images // config.BS) + 1)
    preds = np.argmax(pred_probas, axis=1)

    name_to_idx = testGen.class_indices
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    labels = [idx_to_name[x] for x in testGen.classes]

else:
    for i, img_path in enumerate(imagePaths, start=1):
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        # Only need to expand another dimension if inference on
        #  only one image
        # x = np.expand_dims(x, axis=0)
        label = img_path.split(os.path.sep)[-2]

        data.append(x)
        labels.append(label)

    data = np.array(data)
    # The valAug.mean makes no difference, so no need to preprocess_input
    #  because the mean subtraction seems like actually not applied
    # data = preprocess_input(data)
    print("[INFO] Preprocess completed.")
    print("[INFO] Data shape:", data.shape)

    print("[INFO] Making predictions ...")
    pred_probas = model.predict(data)
    preds = np.argmax(pred_probas, axis=1)

    # print("[INFO] Using ImageDataGenerator with flow method...")
    # valAug = ImageDataGenerator()
    # NOTE: this mean subtraction seems like does not do anything here
    # mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    # valAug.mean = mean
    # testGen = valAug.flow(
    #     x=data,
    #     y=labels,
    #     shuffle=False,
    #     batch_size=config.BS)
    # # Must include this reset method
    # # https://github.com/keras-team/keras/issues/3296
    # testGen.reset()
    # preds = model.predict(testGen,
    #                       steps=(n_images // config.BS) + 1)
    # preds = np.argmax(preds, axis=1)

pred_labels = np.where(preds == 1, "Standing", "NotStanding")
total_correct_preds = np.sum(np.where(pred_labels == labels, True, False))
total_wrong_preds = len(pred_labels) - total_correct_preds
total_time = time.time() - start_time

if FULL_EVAL and total_wrong_preds:
    plot_top_losses(pred_probas, labels, pred_labels, top_k=5)

if PLOT_IMAGES:
    plot_results(imagePaths, labels, pred_labels)

print(f"\n{'Total time elapsed':<26}: {total_time:>8.3f} seconds")
print(f"{'Time elapsed per image':<26}: " f"{(total_time / n_images):>8.3f} seconds\n")
print(f"{'Total images':<26}: {n_images:>8}")
print(f"{'Total correct predictions':<26}: {total_correct_preds:>8}")
print(f"{'Total wrong predictions':<26}: {total_wrong_preds:>8}")

if FULL_EVAL:
    cm = confusion_matrix(labels, pred_labels)

    fig = plt.figure(figsize=(6, 3))
    ax = sns.heatmap(
        cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=5,
        cbar=False,
        yticklabels=["Actual NotStanding", "Actual Standing"],
        xticklabels=["Predicted NotStanding", "Predicted Standing"],
        annot_kws={"fontsize": 13, "fontfamily": "monospace"},
    )

    plt.title("Confusion Matrix", size=15, fontfamily="serif")
    plt.savefig(os.path.join(config.OUTPUTS_PATH, "confusion_matrix.png"))
    plt.show()
