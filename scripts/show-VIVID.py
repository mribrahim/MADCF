import os
import cv2
import json
import numpy as np
import argparse


def readBoxes(path):
    boxes = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            x,y,w,h = line.split(',')
            if "NaN" == x:
                boxes.append([])
            else:
                boxes.append([int(x), int(y), int(w), int(h)])
    return boxes

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', "-s", type=str, default='egtest01', help='sequence (egtest01, egtest02...)') 
parser.add_argument('--path', "-p", type=str, default="/home/ibrahim/Desktop/Dataset/UAV VIVID Tracking Evaluation/", help='VIVID dataset path') 

args = parser.parse_args()


pathImg =  args.path + args.sequence + "/" + args.sequence + "/"
pathGT =  "../VIVID/" + args.sequence + "/result-" + args.sequence + "-GT.txt"



file_list = sorted(os.listdir(pathImg))
boxesGT = readBoxes(pathGT)
counter = 0
for filename in file_list:
    img = cv2.imread(pathImg + filename)

    x,y,w,h = boxesGT[counter]
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2, 1)

    counter += 1
    cv2.imshow("img", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()