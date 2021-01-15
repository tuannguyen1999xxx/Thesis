import cv2
import numpy as np
import os

file = open("results.txt","r")
lines = file.readlines()
print(lines)
infors = [x[0:-12] for x in lines]
print(infors)
list_infor = []
for infor in infors:
    infor_need = infor.split(",")
    list_infor.append(infor_need)
print(list_infor)
path = "C:/Users/Windows 10/PycharmProjects/test/FairMOT-master/results/frame"
path_peron = "C:/Users/Windows 10/PycharmProjects/test/FairMOT-master/results/person"
if (os.path.exists(path_peron) == False):
    os.mkdir(path_peron)
for x in list_infor:
    frame_id,id_person = int(x[0]),x[1]
    path1 = path_peron + '/' + str(id_person)

    if(os.path.exists(path1) == False):
        os.mkdir(path1)

    x,y,w,h = int(float(x[2])),int(float(x[3])),int(float(x[4])),int(float(x[5]))
    img = cv2.imread(path + '/' + '{:05d}.jpg'.format(frame_id))
    if (img is not None):
        cv2.imwrite(path1 + '/' + str(id_person)+'-' +str(x) +'.jpg',img[y:y+h,x:x+w])
file.close()