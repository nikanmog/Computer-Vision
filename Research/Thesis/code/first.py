import glob
import shutil
import os
src_dir = "images/train/images"
dst_dir = "images/train/"
for jpgfile in glob.iglob(os.path.join(src_dir, "**", "*.jpg")):
    shutil.move(jpgfile, dst_dir) 
src_dir = "images/test/images"
dst_dir = "images/test/"
for jpgfile in glob.iglob(os.path.join(src_dir, "**", "*.jpg")):
    shutil.move(jpgfile, dst_dir) 
    
src_dir = "images/train/images/annotations"
dst_dir = "images/train/"
for xmlfile in glob.iglob(os.path.join(src_dir, "**", "*.xml")):
    shutil.move(xmlfile, dst_dir) 
src_dir = "images/test/images/annotations"
dst_dir = "images/test/"
for xmlfile in glob.iglob(os.path.join(src_dir, "**", "*.xml")):
    shutil.move(xmlfile, dst_dir) 