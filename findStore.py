from sklearn.metrics.pairwise import pairwise_distances  # Compare feature vector by distance
import cv2
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
from findmoment import moment

fig = plt.figure(figsize=(5, 5))
fig2 = plt.figure(figsize=(5, 5))
colums = 2
rows = 2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
args = vars(ap.parse_args())

imagePaths = sorted(glob.glob(args["dataset"] + "/*.jpg"))
data = []
inputm = []

inputIm = cv2.imread("test.jpg")  # input image here
inputm.append(moment(inputIm))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    data.append(moment(image))

D = pairwise_distances(data, inputm)  # calculate Euclidean distance between each paired vectors
print("Data moment: ", data)
print("Input moment: ", inputm)
i = np.argpartition(D, 4, None)[:4]  # get the 4 minimum distances index
x = 1;
plt.figure(1)
# print("i: ", i)
for i1 in i:
    image = cv2.imread(imagePaths[i1])
    print("Found something: {}".format(imagePaths[i1]))
    fig.add_subplot(rows, colums, x).set_title(str(D[i1]))
    plt.imshow(image)
    x = x + 1
plt.figure(2)
fig2.add_subplot(1, 1, 1).set_title("Original")
plt.imshow(inputIm)
plt.show()
