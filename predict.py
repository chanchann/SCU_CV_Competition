# -*- coding: UTF-8 -*-
import numpy as np
import torch
import json
import one_hot_encoding
from torch.autograd import Variable
import torchvision.models as model
#from visdom import Visdom # pip install Visdom
import setting
import my_dataset
from cnn_model import CNN
dest_path='./result.json'
def main():
    cnn = model.resnet50(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    cnn.eval()
    cnn.load_state_dict(torch.load('./checkpoints/resnet50_modelbest.pkl', map_location='cpu'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()
    res = []
    cnt=0
    #vis = Visdom()
    for i, (images, labels,image_root) in enumerate(predict_dataloader):
        image = images
        # print(i)
        # print(labels)
        # print(images)
        # print(image_root)
        image_root=image_root[0]
        id=image_root[-10:-5]+image_root[-4:]
        vimage = Variable(image)
        predict_label = cnn(vimage)
        c0 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        cnt+=1
        print(cnt)
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        # print(predict_label)
        #vis.images(image, opts=dict(caption=c))
        dict_map={}
        dict_map['id']=id
        dict_map['characters']=predict_label
        res.append(dict_map)
        # print(res)

    with open(dest_path,'w') as f:
        f.write(json.dumps(res))
if __name__ == '__main__':
    main()
