# Authors : Bianchi Alexandre and Moulin Vincent
# Class : INF3dlma
# HE-Arc 2019-2020
# PedestriArc : Detection of pedestrian crossing

# Import
from PIL import Image
import glob

# Functions
def readData(foldername):
    image_list = []
    for filename in glob.glob(foldername + '/*.jpg'):
        im=Image.open(filename)
        image_list.append(im)
    return image_list

# Main
if __name__ == "__main__":
    foldername = "data"
    image_list = readData(foldername)
    for img in image_list:
        print(img.filename)






    