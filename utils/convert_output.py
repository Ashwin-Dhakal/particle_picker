import cv2
import pandas as pd
import mrcfile
from matplotlib import pyplot as plt, patches
from PIL import Image
import numpy as np


def draw_bounding_box(image, boxes):
    # load image
    # image = cv2.imread("10_data/10345_10_data/micrographs/18jam15a_0007_ali_DW.mrc")
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in boxes:
        rec = patches.Rectangle((box[0], box[1]), box[2], box[2], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rec)

    plt.imshow(img, cmap='gray')
    plt.show()


# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     rect = cv2.boundingRect(c)
#     if rect[2] < 100 or rect[3] < 100: continue
#     print cv2.contourArea(c)
#     x,y,w,h = rect
#     cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
# cv2.imshow("Show",im)
# cv2.waitKey()
# cv2.destroyAllWindows()


data = pd.read_csv("10_data/10345_10_data/particle_coordinates/18jam15a_0007_ali_DW.csv")
data = data[['X-Coordinate', 'Y-Coordinate', 'Diameter']]

data['x'] = data['X-Coordinate'] - data['Diameter'] / 2
data['y'] = data['Y-Coordinate'] - data['Diameter'] / 2

data['width'] = data['Diameter']

data = data[['x', 'y', 'width']]

boxes = list(data.itertuples(index=False, name=None))

# from PIL import Image
# img = Image.open("10_data/10345_10_data/micrographs/18jam15a_0009_ali_DW.mrc")#.convert('L')
# print(img)
#
# exit()
with mrcfile.open("10_data/10345_10_data/micrographs/18jam15a_0007_ali_DW.mrc") as mrc:
    print(mrc.data.shape)
    img = Image.fromarray(np.uint8(mrc.data)).convert('L')

# im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
# grayImage = uint8(255 * mat2gray(originalImage)); imshow(grayImage);


draw_bounding_box(img, boxes)
