from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import color
import argparse
import pandas as pd
import skimage
import random
import os
from pathlib import Path

class ImageSegmentation:
    def __init__(self, original_image, img_name, k_value, n_segment, color_types):
        self.k_value = k_value
        self.img_name = img_name
        self.colors = self.generate_colors(color_types, self.k_value)
        self.original_image = original_image
        self.height, self.width, self.channels = self.original_image.shape
        self.segments_slic = slic(self.original_image, n_segments=n_segment, compactness=10)
        self.superpixels = mark_boundaries(self.original_image, self.segments_slic)
        self.segments_slic = self.segments_slic.reshape(-1)
        self.R, self.G, self.B = cv2.split(self.original_image)

        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "results")):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "results"))


    def kmeans(self, X, k, max_iters=100):
        np.random.seed(123)
        centroids = np.array(X[np.random.choice(X.shape[0], k, replace=False)])
        
        for _ in range(max_iters):
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        return labels
    
    def min_max_norm(self, list):
        max_val = max(list)
        min_val = min(list)

        return (list - min_val) / (max_val - min_val)
    
    def generate_colors(self, color_types, k_value):
        colors = list()
        np.random.seed(1)
        for i in np.random.randint(0, 16, k_value):
            colors.append(color_types[i][1])
        
        return np.array(colors)


    def saving_segmentations(self, seg_type, n_segment, result_type, image):
        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name)):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name))

        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, )):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, ))

        if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value))):
            os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value)))

        if n_segment==None:
            matplotlib.image.imsave(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value), result_type + ".jpg"), image)
        
        else:
            if not os.path.exists(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value), "n_segment="+str(n_segment))):
                os.makedirs(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value), "n_segment="+str(n_segment)))

            matplotlib.image.imsave(os.path.join(Path(__file__).resolve().parents[1], "results", self.img_name, seg_type, "k_value=" + str(self.k_value), "n_segment="+str(n_segment), result_type + ".jpg"), image)
    
    def rgb_color(self):
        
        concatenated_pixels = self.original_image.reshape(-1, self.channels)
        vectorized_pixels = np.float32(concatenated_pixels)
        labels = self.kmeans(vectorized_pixels, self.k_value)
        res = self.colors[labels]
        image_labels = res.reshape((self.original_image.shape))
        image_segments = mark_boundaries(self.original_image, labels.reshape(self.original_image.shape[0:2]))

        self.saving_segmentations(seg_type = "pixel_rgb-color", n_segment=None, result_type = "image_labels", image = image_labels.astype(np.uint8))
        self.saving_segmentations(seg_type = "pixel_rgb-color", n_segment=None, result_type = "image_segments", image = image_segments)

    def rgb_color_location(self):
        pixels_with_coordinates = []
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = self.original_image[y, x]
                pixels_with_coordinates.append([r, g, b, y, x])

        pixels_with_coordinates = np.array(pixels_with_coordinates, dtype=float).T

        for feature_idx in range(len(pixels_with_coordinates)):
            pixels_with_coordinates[feature_idx] = self.min_max_norm(pixels_with_coordinates[feature_idx])

        pixels_with_coordinates = pixels_with_coordinates.T

        vectorized_pixels = np.float32(pixels_with_coordinates)
        labels = self.kmeans(vectorized_pixels,self.k_value)
        res = self.colors[labels]
        image_labels = res.reshape((self.original_image.shape))
        image_segments = mark_boundaries(self.original_image, labels.reshape(self.original_image.shape[0:2]))

        self.saving_segmentations(seg_type = "pixel_rgb-color-location", n_segment=None, result_type = "image_labels", image = image_labels.astype(np.uint8))
        self.saving_segmentations(seg_type = "pixel_rgb-color-location", n_segment=None, result_type = "image_segments", image = image_segments)

    def mean_rgb_color(self):

        R, G, B = cv2.split(self.original_image)
        R, G, B = R.reshape(-1), G.reshape(-1), B.reshape(-1)

        data_tuples = list(zip(R,G,B, self.segments_slic))
        df = pd.DataFrame(data_tuples, columns=['Red',"Green","Blue",'segments_slic'])
        df_group = df.groupby("segments_slic")
        df_group_mean = df_group.mean()
        mean_RGB = df_group_mean.to_numpy()

        vectorized_pixels = np.float32(mean_RGB)
        extended_labels = list()
        labels = self.kmeans(vectorized_pixels, self.k_value)

        for i in range(self.segments_slic.shape[0]):
            extended_labels.append(labels[(df["segments_slic"][i])-1])

        extended_labels = np.array(extended_labels)
        res = self.colors[extended_labels]
        image_labels = res.reshape((self.original_image.shape))
        image_segments = mark_boundaries(self.original_image, extended_labels.reshape(self.original_image.shape[0:2]))

        self.saving_segmentations(seg_type = "superpixel_mean-rgb-color", n_segment=df_group.mean().shape[0], result_type = "image_labels", image = image_labels.astype(np.uint8))
        self.saving_segmentations(seg_type = "superpixel_mean-rgb-color", n_segment=df_group.mean().shape[0], result_type = "image_segments", image = image_segments)
        self.saving_segmentations(seg_type = "superpixel_mean-rgb-color", n_segment=df_group.mean().shape[0], result_type = "superpixels", image = self.superpixels)
        

    def rgb_color_histogram(self):

        R, G, B = cv2.split(self.original_image)
        R, G, B = R.reshape(-1), G.reshape(-1), B.reshape(-1)

        data_tuples = list(zip(R,G,B, self.segments_slic))
        df = pd.DataFrame(data_tuples, columns=['Red',"Green","Blue",'segments_slic'])
        df_group = df.groupby("segments_slic")

        bins = range(0,257)
        superpixel_hist = list()
        for i in range(1,df_group.mean().shape[0]+1):
            R, G, B = df_group.get_group(i)["Red"],df_group.get_group(i)["Green"],df_group.get_group(i)["Blue"]
            hist_r, bins = np.histogram(R, bins)
            hist_g, bins = np.histogram(G, bins)
            hist_b, bins = np.histogram(B, bins)
            superpixel_hist.append(np.array(list(zip(hist_r, hist_g, hist_b))))
        superpixel_hist = np.array(superpixel_hist).reshape(df_group.mean().shape[0], -1)

        vectorized_pixels = np.float32(superpixel_hist)
        extended_labels = list()
        labels = self.kmeans(vectorized_pixels,self.k_value)

        for i in range(self.segments_slic.shape[0]):
            extended_labels.append(labels[(df["segments_slic"][i])-1])
            
        extended_labels = np.array(extended_labels)
        res = self.colors[extended_labels]
        image_labels = res.reshape((self.original_image.shape))
        image_segments = mark_boundaries(self.original_image, extended_labels.reshape(self.original_image.shape[0:2]))

        self.saving_segmentations(seg_type = "superpixel_rgb-color-histogram", n_segment=df_group.mean().shape[0], result_type = "image_labels", image = image_labels.astype(np.uint8))
        self.saving_segmentations(seg_type = "superpixel_rgb-color-histogram", n_segment=df_group.mean().shape[0], result_type = "image_segments", image = image_segments)
        self.saving_segmentations(seg_type = "superpixel_rgb-color-histogram", n_segment=df_group.mean().shape[0], result_type = "superpixels", image = self.superpixels)

    def mean_gabor(self):

        filt_realR, filt_realG, filt_realB = np.zeros((self.original_image.shape[0], self.original_image.shape[1])), np.zeros((self.original_image.shape[0], self.original_image.shape[1])), np.zeros((self.original_image.shape[0], self.original_image.shape[1]))
        for i in range(12):
            for j in range(1,10):
                realR, _ = skimage.filters.gabor(self.R, frequency=j*0.1, theta=i*30.0)
                realG, _ = skimage.filters.gabor(self.G, frequency=j*0.1, theta=i*30.0)
                realB, _ = skimage.filters.gabor(self.B, frequency=j*0.1, theta=i*30.0)

                filt_realR += realR
                filt_realG += realG
                filt_realB += realB

        filt_realR, filt_realG, filt_realB = (filt_realR/110).reshape(-1), (filt_realG/110).reshape(-1), (filt_realB/110).reshape(-1)

        data_tuples = list(zip(filt_realR,filt_realG,filt_realB, self.segments_slic))
        df = pd.DataFrame(data_tuples, columns=['Red',"Green","Blue",'segments_slic'])
        df_group = df.groupby("segments_slic")
        mean_gabor_RGB = df_group.mean().to_numpy()

        vectorized_pixels = np.float32(mean_gabor_RGB)
        extended_labels = list()
        #buraya time ekle
        labels = self.kmeans(vectorized_pixels,self.k_value)

        for i in range(self.segments_slic.shape[0]):
            extended_labels.append(labels[(df["segments_slic"][i])-1])

        extended_labels = np.array(extended_labels)
        res = self.colors[extended_labels]
        image_labels = res.reshape((self.original_image.shape))
        image_segments = mark_boundaries(self.original_image, extended_labels.reshape(self.original_image.shape[0:2]))

        self.saving_segmentations(seg_type = "superpixel_mean-gabor", n_segment=df_group.mean().shape[0], result_type = "image_labels", image = image_labels.astype(np.uint8))
        self.saving_segmentations(seg_type = "superpixel_mean-gabor", n_segment=df_group.mean().shape[0], result_type = "image_segments", image = image_segments)
        self.saving_segmentations(seg_type = "superpixel_mean-gabor", n_segment=df_group.mean().shape[0], result_type = "superpixels", image = self.superpixels)


color_types = [
    ("black", (0, 0, 0)),
    ("silver", (192, 192, 192)),
    ("gray", (128, 128, 128)),
    ("white", (255, 255, 255)),
    ("maroon", (128, 0, 0)),
    ("red", (255, 0, 0)),
    ("purple", (128, 0, 128)),
    ("fuchsia", (255, 0, 255)),
    ("green", (0, 128, 0)),
    ("lime", (0, 255, 0)),
    ("olive", (128, 128, 0)),
    ("yellow", (255, 255, 0)),
    ("navy", (0, 0, 128)),
    ("blue", (0, 0, 255)),
    ("teal", (0, 128, 128)),
    ("aqua", (0, 255, 255))
]

k_value = 2
n_segment = 100
img_name = "deer.jpg" # flower.jpg, human.jpg, ladybird.jpg, pyramid.jpg, horses.jpg, short_tree.jpg


image = cv2.cvtColor(cv2.imread(os.path.join(Path(__file__).resolve().parents[1],"input_images", img_name)), cv2.COLOR_BGR2RGB)

image_segmentation = ImageSegmentation(image, img_name[:-4], k_value, n_segment, color_types)
image_segmentation.rgb_color()
image_segmentation.rgb_color_location()
image_segmentation.mean_rgb_color()
image_segmentation.rgb_color_histogram()
image_segmentation.mean_gabor()