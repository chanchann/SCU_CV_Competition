# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as model 
import setting
import my_dataset
from cnn_model import CNN
import one_hot_encoding
from densenet import DenseNet

def main():
    #cnn = CNN()
    #cnn = DenseNet(depth=16, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    cnn = model.resnet50(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    cnn.eval()
    cnn.load_state_dict(torch.load('./checkpoints/resnet50_modelbest.pkl', map_location='cpu'))
    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels,_) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:setting.ALL_CHAR_SET_LEN].data.numpy())]
#<<<<<<< Updated upstream
#        c1 = setting.ALL_CHAR_SET[np.argmax(predict_label[0,setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
#=======
        c1 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
#>>>>>>> Stashed changes
        c2 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        # print("predict: {},   true: {}".format(predict_label, true_label))       
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
        else:
            print("predict: {},   true: {}".format(predict_label, true_label))
        if(total%200==0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()
