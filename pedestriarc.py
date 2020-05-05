'''
 Authors : Bianchi Alexandre and Moulin Vincent
 Class : INF3dlma
 HE-Arc 2019-2020
 PedestriArc : Detection of pedestrian crossing
'''

# -------- #
# Imports
# -------- #

import glob
import json
import os
import pandas

from zipfile import ZipFile
from crosswalk_detection_with_class import process
from shutil import copy
from shutil import rmtree
from PIL import Image

# -------- #
# Functions
# -------- #

def readData(foldername):
    '''
    Parameter : foldername = string who match a folder at the root of script
    Return : image_list = list who contains file name of all .jpg and .png file
    Description : Creates a list who contains all file name ended by jpg or png and returns the list
    '''
    image_list = []

    types = ('*.jpg', '*.png') # the tuple of file types
    files_grabbed = []

    for files in types:
        files_grabbed.extend(glob.glob(foldername + "/" + files))

    for filename in files_grabbed:
        im=Image.open(filename)
        image_list.append(im)

    return image_list

# -------- #
# Main
# -------- #

if __name__ == "__main__":
    foldername = "data"
    image_list = readData(foldername)
    result = []

    # Create the path where files with crosswalk will be copied
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data_filtered"
    folder_path =  r'data_filtered' 
    path = os.path.join(dir_path, folder_path, "") 

    # remove folder if one is already exist
    if os.path.exists(path):
        rmtree(path)
        print("clean directory done!")
    
    # create a new empty folder to reiceive the file
    os.makedirs(path)
    print("data_filtered folder has been successfuly created!")

    for img in image_list:

        filename = img.filename.split('\\')

        res = process("data\\" + filename[1])
        
        if res is None:
            print("no cross walk in " + filename[1])
            containCrosswalk = False
            roi = {}
        else:
            print("cross walk in " + filename[1])
            containCrosswalk = True
            #roi = {'x':[str(res[0])], 'y':[str(res[1])]}
            roi = {'leftCorner': { 'x':int(res[0][0]), 'y':int(res[0][1])},
            'rightCorner': { 'x':int(res[1][0]), 'y':int(res[1][1])}}

            # if file has a crosswalk, we copy it to the folder
            copy(str(img.filename), path)
            print(filename[1] + " has been successfully copied!")

        # Add dictionnary with information about picture on the result list
        result.append({"file": filename[1], "crosswalk": containCrosswalk, "ROI": roi})

    # Print the list result in a json format inside a file data.json
    with open('data_filtered/data_filtered.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        print("data_filtered.json file has been successfuly saved!")

    # pandas dataframe creation for export to csv 
    df = pandas.DataFrame(result)

    # export data to csv file
    df.to_csv("data_filtered/data_filtered.csv", sep=',', index=False)
    print("data_filtered.csv file has been successfuly saved!")    

    # writing files to a zipfile
    with ZipFile('data_filtered.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk("data_filtered"):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath)
    print("data_filtered.zip folder has been successfuly created!") 

    # remove the no zip folder 
    if os.path.exists(dir_path):
        rmtree(dir_path)
        print("clean folder data_filtered done!")