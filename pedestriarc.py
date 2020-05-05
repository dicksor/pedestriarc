'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing
'''

# Import
from PIL import Image
import glob
import json
from processImage import findCrosswalk

# Functions

def readData(foldername):
    '''
    Parameter : foldername = string who match a folder at the root of script
    Return : image_list = list who contains file name of all .jpg file
    Description : Creates a list who contains all file name ended by jpg and returns the list
    '''
    image_list = []
    for filename in glob.glob(foldername + '/*.jpg'):
        im=Image.open(filename)
        image_list.append(im)
    return image_list

# Main
if __name__ == "__main__":
    foldername = "data"
    image_list = readData(foldername)
    result = []
    for img in image_list:
        containCrosswalk = True
        roi = [15,30]
        # Add dictionnary with information about picture on the result list
        result.append({"file": img.filename, "crosswalk": containCrosswalk, "ROI": roi})

    # Print the list result in a json format inside a file data.json
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("data.json file has been successfuly saved!")

    for img in image_list:
        findCrosswalk(img.filename)





    