import os
import cv2
import json
import numpy as np

import xml.etree.ElementTree as ET

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        # filename = root.find('filename').text
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(float(boxes.find("bndbox/ymin").text))
        xmin = int(float(boxes.find("bndbox/xmin").text))
        ymax = int(float(boxes.find("bndbox/ymax").text))
        xmax = int(float(boxes.find("bndbox/xmax").text))

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


sequence = "Pexels-Zaborski"


## PATH to the PESMOD dataset:  download from https://github.com/mribrahim/PESMOD
pathImg =  "/home/ibrahim/Desktop/Dataset/my IHA dataset/PESMOD/" + sequence + "/images/"
pathGT =  "../PESMOD/" + sequence + "/annotations/"



file_list = sorted(os.listdir(pathImg))

counter = 0
totalTP, totalFP, totalTN, totalFN, totalIOU = 0, 0, 0, 0, 0

for filename in file_list:
    img = cv2.imread(pathImg + filename)

    xml_path = pathGT + filename.replace(".jpg", ".xml")
    if os.path.exists(xml_path):
        gt_box = read_content(xml_path)
        for x1,y1,x2,y2 in gt_box:

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2, 1)

    img = cv2.resize(img, (1280, 720))
    cv2.imshow("img", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()