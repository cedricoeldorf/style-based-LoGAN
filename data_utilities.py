""" data utilities """
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

""" List images in certain directory """
def list_filenames(mypath):

  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  return onlyfiles

""" Load images into memory and size specification """
def load_images(path, filenames, resolution,mnist = True):
    # Specify Size
    df = []
    for file in filenames:
        if mnist == True:

            img = cv2.imread(path + file, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        print('Original Dimensions : ',img.shape)

        dim = (resolution, resolution)
        # resize image
        resized = cv2.resize(img, dim)
        if mnist == True:
            resized = np.ravel(resized)
        resized = resized / 255
        print('New Dimensions : ',resized.shape)
        df.append(resized)
    return df
