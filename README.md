# OCR

The first SCU AI image competition.

## File

## 简要说明

clone下来后先``python train.py``测试

### 完整使用说明

先用``json_help.py``生产新的``train_data.json``文件方便label操作

再用``process_img_name.py``将图片名修改

``mv_img.py``选择你移动的数量，来分离train/test

根据以上几步将你的完整数据集化为仓库中dataset的形式

-----------------


``train.py``来训练你的模型

``test.py``评估模型，如果效果不好，修改``cnn_model.py``,然后再次训练``train.py``

效果良好，进行``predict.py``


###  code说明

``cnn_model.py``:搭建模型

``img_cnt.py``:数文件夹中图片个数的脚本，可一键看dataset中train,test,predict的图片数量

``json_help.py``:封装了一些对于json文件的操作，

将{"id":"15999.jpg","characters":"uFtN"}

转为为｛"49998.jpg": "gxMz"｝

方便label

存为train_data.json

``mv_img.py``:批量移动train的一部分至test,改变num即可

``my_dataset.py``:load data

``one-hot-encoding.py``：独热向量编码

``predict.py``:对于predict中进行预测

``process_img_name``：我们将label加到name上

如'00153.jpg'->'WxaY_00153.jpg'

``setting.img``:一些设置

``test.py``:用train保存的pkl对测试集进行模型评估

``train.py``：训练模型，保存pkl


- src

文件中皆为测试代码

- dataset

目前放入一张图片观察效果，请本地下载pan中数据集train and test

然后我把test->predict

重新创建了一个test文件来分离train 为训练集和测试集


## 基本步骤：

- 准备原始图片素材
- 图片预处理
- 图片字符切割
- 图片尺寸归一化
- 图片字符标记
- 字符图片特征提取
- 生成特征和标记对应的训练数据集
- 训练特征标记数据生成识别模型
- 使用识别模型预测新的未知图片集
- 达到根据“图片”就能返回识别正确的字符集的目标

### 原始图片素材

已提供5w train 和5w test

train_annotation.json中定义了图片：lable

### 图片预处理

- 读取原始图片素材
- 将彩色图片二值化为黑白图片
- 去除背景噪点

### TODO

- 选择模型

- 神经网络代码构建

- 测试评估



