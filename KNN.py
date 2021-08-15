# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:07:03 2021

@author: Lital Barak and Avishai Wizel
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt



def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

#load data and labels from path
class SimpleDatasetLoader:
	def load(self, imagePaths):
		# initialize the list of features and labels
		data = []
		labels = []
		# loop over the input images
		import os
		from glob import glob
		all_images_paths =  [y for x in os.walk(imagePaths) for y in glob(os.path.join(x[0], '*.jpeg'))]
		for (i, imagePath) in enumerate(all_images_paths):
			image = cv2.imread(imagePath)
			label = imagePath.split('\\')[-2]
            
			image = image_to_feature_vector(image)
			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)
		return (np.array(data), np.array(labels))


#%% main

#load data
trainPath = "C:\\Users\\avish\\CLionProjects\\IP data\\OCT2017\\OCT2017\\train"
sdl_train = SimpleDatasetLoader()
(trainX, trainY_original) = sdl_train.load(trainPath)

testPath = "C:\\Users\\avish\\CLionProjects\\IP data\\OCT2017\\OCT2017\\test"
sdl_test = SimpleDatasetLoader()
(testX, testY_original) = sdl_test.load(testPath)

# encode text labels to numeric labels
le = LabelEncoder()
trainY = le.fit_transform(trainY_original)
testY = le.fit_transform(testY_original)

#create and fit model
model = KNeighborsClassifier(n_neighbors=1)
model = model.fit(trainX, trainY)

# get predictions
y_pred = model.predict(testX)

#calculate accuracy
acc = model.score(testX, testY) 
print ("accuracy=", acc)

from sklearn.metrics import f1_score
f1_score(testY,y_pred,average = 'weighted')
#%% Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testY, y_pred)

#%% ROC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


#create one hot encoding labels
y_test = label_binarize(testY, classes=[0,1,2,3])
y_score = label_binarize(y_pred, classes=[0,1,2,3])
n_classes = 4

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + le.classes_[i])
    plt.legend(loc="lower right")
    plt.show()





#%% compute_class_weight
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0,1,2,3])
y =  np.array (trainY)
class_weight_train = 1/compute_class_weight("balanced", classes, y)/3

y =  np.array (testY)
class_weight_test = 1/compute_class_weight("balanced", classes, y)

