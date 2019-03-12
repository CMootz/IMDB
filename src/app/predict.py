import torch
import torch.nn as nn
import imageio
from skimage import color
from skimage.transform import rescale, resize
import numpy as np


class Predict:
    def __init__(self, path):
        self.load_model = torch.load('../../model/softmax')
        self.model = self.load_model.eval()
        self.data_img = self.load_image(path)
        self.y_pred = self.predict(self.data_img)

    def predict(self, data):
        linears = self.model(data)
        agg_model = nn.Softmax()
        predict = agg_model(linears)

        _, max_indices = predict.max(1)
        return max_indices

    def load_image(self, image_file, my_scale=(16, 16)):
        im = imageio.imread(image_file)
        im = color.rgb2gray(im)
        resized = resize(im, my_scale)
        resized *= -1
        resized += 0.5
        resized /= np.abs(resized).max()
        tens = torch.as_tensor(resized.reshape(1, 256), device='cpu', dtype=torch.float)
        return tens
