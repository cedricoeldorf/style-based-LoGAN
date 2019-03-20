""" data utilities """
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import pandas as pd
""" List images in certain directory """

## Load Training set, Test set (both from separate folders)
## Within testing df are the targets and labels
def list_filenames(mypath,test_path):

  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

  df = pd.read_csv(test_path)
  print(df.columns)
  test_files = list(df.filenames)
  targets = list(df.label)
  return onlyfiles, test_files, targets

""" Load images into memory and size specification """
## Normalization being done

def load_images(path, filenames, resolution,mnist = True):
    # Specify Size
    df = []
    for file in filenames:
        if mnist == True:

            img = cv2.imread(path + file, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        print('Original Dimensions : ',img.shape)

        # Some images have 4th alpha channel, don't use these.
        if img.shape[2] == 3:

            dim = (resolution, resolution)
            # resize image
            resized = cv2.resize(img, dim)
            if mnist == True:
                resized = np.ravel(resized)
            resized = resized / 255
            print('New Dimensions : ',resized.shape)
            df.append(resized)
    return df
