# -*- coding: UTF-8 -*-
import json
import sys
import time
import os
import glob as gb
from json_help import load
path='./dataset/predict/*.jpg'
img_path=gb.glob(path)
time.clock()
for path in img_path:
    print(path)
    path=os.path.split(path)
    os.rename(path[0]+'/'+path[1],path[0]+'/'+path[1][:-4]+'_'+path[1][-4:])
print('总共运行时间:%0.2f'%(time.clock()))

