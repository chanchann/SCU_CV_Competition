import os
import shutil
import glob as gb

dir_src='./train/*.jpg'
dir_dest='./test/'

def mv_img(num,src,dest):
    img_path=gb.glob(src)
    cnt=1
    for path in img_path:
        path_tuple=os.path.split(path)
        shutil.move(path,dest+path_tuple[1])
        if(cnt<num):
            cnt+=1
        else:
            break

if __name__=='__main__':
    mv_img(10000,dir_src,dir_dest)