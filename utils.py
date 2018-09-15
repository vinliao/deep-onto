import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from definitions import ROOT_DIRECTORY

image_label_list = [] #list of category names to avoid confusion

def get_image_data(image_size=224):
    image_category = []
    image_path = os.path.join(ROOT_DIRECTORY, 'data', 'Images')
    model_path = os.path.join(ROOT_DIRECTORY, 'data', 'models')
    category_path = model_path + '/image_category.pkl'

    total_image_file = 0
    count = 0

    #count the total image file in the /data/image folder
    for root, dirs, files in os.walk(image_path):
        for _ in files:
            total_image_file+=1

    x = np.zeros((total_image_file, image_size, image_size, 3), dtype=np.uint8)
    target_list = []

    #===reading images into image_list array===
    category = os.listdir(image_path)
    for i, cat in enumerate(category):
        img_list = os.listdir(image_path + '/{}'.format(cat))
        image_label_list.append(cat)
        for image_name in img_list:
            #insert the image into np array
            x[count, :] = cv2.resize(cv2.imread('%s/%s/%s' %(image_path, cat, image_name)),\
                    (image_size, image_size))
            count+=1

            #create the categorical target list
            #e.g. agricultural: 1, airplane: 2, beach: 3, ...
            target_list.append(i+1)
        image_category.append(str(cat))

    #===create one hot vector===
    y = np.zeros((total_image_file, len(image_label_list)), dtype=np.int32)
    for i, target in enumerate(target_list):
        #from the zero array, set the value of the corresponding index to 1
        y[i][target-1] = 1

    #pickle the target data so that it can be used on inference
    with open(category_path, 'wb') as f:
        pickle.dump(image_category, f)

    #===splitting data===
    #train/valid/test = 70/15/15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    return x_train, x_valid, x_test, y_train, y_valid, y_test
