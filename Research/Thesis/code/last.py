import csv
import glob
import shutil
import os

src_dir = "images/train/"
dst_dir = "C:/tensorflow1/models/research/object_detection/images/train"

with open('images/train_labels.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0
    for row in spamreader:       
        if count != 0:
            shutil.copy(os.path.join(src_dir,row[0]), dst_dir) 
        count = 1

src_dir = "images/test/"
dst_dir = "C:/tensorflow1/models/research/object_detection/images/test"

with open('images/test_labels.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0
    for row in spamreader:       
        if count != 0:
            shutil.copy(os.path.join(src_dir,row[0]), dst_dir) 
        count = 1