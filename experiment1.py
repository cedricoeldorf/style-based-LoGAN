""" Experiment 1: Test computational time for different resolutions """

resolution = 32
visualize = False
""" Libraries """
import cv2
import pandas as pd
import numpy as np
import pickle
import csv
from data_utilities import list_filenames, load_images
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.style.use(['qb-common', 'qb-light'])

# Get filenames
path = "./Data/LLD-logo_sample/"
filenames = list_filenames(path)

# All images within dataframe
df = load_images(path, filenames, resolution)
del df[156]
df_test = df[-10:]
df = df[0:-10]
# based on resolution, run and time autoencoder train

from ae_builder import *

autoencoder = AE(resolution, df, df_test,0,0,1)

import time

start = time.time()
autoencoder.model()
end = time.time()
print(end - start)
to_save = [resolution, end - start]
with open('./experiment_1_results/times.csv','a') as fd:
    writer = csv.writer(fd)
    writer.writerow(to_save)


if visualize == True:

    pl = pd.read_csv("./experiment_1_results/times.csv")

    x = pl.Resolution
    y = pl.Time

    def exponenial_func(x, a, b, c):
        return a*np.exp(-b*x)+c

    popt, pcov = curve_fit(exponenial_func, x, y, p0=(1, 1e-6, 1))
    xx = np.linspace(0,400)
    yy = exponenial_func(xx, *popt)
    plt.plot(x,y,"o",xx,yy)
    plt.title('Correlation: Training Time and Image Resolution')
    plt.ylabel('Training Time (in seconds)')
    plt.xlabel('Image Resolution')
    plt.show()
