import glob as gb

train_path='./dataset/train/*.jpg'
test_path='./dataset/test/*.jpg'
predict_path='./dataset/predict/*.jpg'

def count_img(path):
    img_path=gb.glob(path)
    cnt=0
    for i in img_path:
        cnt+=1
    return cnt

if __name__=='__main__':
    print('Traing:'+str(count_img(train_path)))
    print('Test:'+str(count_img(test_path)))
    print('Predict:'+str(count_img(predict_path)))