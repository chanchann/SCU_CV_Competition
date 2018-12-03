# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torchvision.models as model
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import my_dataset
from cnn_model import CNN
import setting
from densenet import DenseNet

# Hyper Parameters
num_epochs = 100 #30
batch_size = 64 #100
learning_rate = 0.001

def main():
    #cnn = CNN()
    #cnn = DenseNet(depth=16, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    #cnn = model.vgg11_bn(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    cnn = model.resnet50(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    #cnn = model.inception_v3(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    #cnn = model.squeezenet1_1(pretrained=False, num_classes=setting.MAX_CAPTCHA*setting.ALL_CHAR_SET_LEN)
    cnn.train()
    cnn.cuda()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    #cnn.load_state_dict(torch.load('./checkpoints/densenet_model.pkl'))
    print('init net')

    best_loss = 100
    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        scheduler.step()
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).cuda()
            labels = Variable(labels.float()).cuda()
            optimizer.zero_grad()
            predict_labels = cnn(images)
            #print(predict_labels.size())
            #print(labels.size())
            loss = criterion(predict_labels, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                cur_loss = loss.item()
                print("epoch:", epoch, "step:", i, "loss:", cur_loss)
                if cur_loss < best_loss:
                    best_loss = cur_loss
            #if (i+1) % 50 == 0 and loss.item() <= best_loss:
                    torch.save(cnn.state_dict(), "./checkpoints/resnet50_modelbest.pkl")   #current is model.pkl
                    print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    if loss.item() <= best_loss:
        torch.save(cnn.state_dict(), "./checkpoints/resnet50_modelbest.pkl")   #current is model.pkl
        print("save last model")
    print("done!")

if __name__ == '__main__':
    main()
