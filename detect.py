from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
    
args = arg_parse() # 加载调用参数，所有参数都有默认值
images = args.images # 图像存储的目录
batch_size = int(args.bs) # batch size，默认为1，在命令行中添加 -bs n 确定batch size大小
confidence = float(args.confidence) # 置信度
nms_thesh = float(args.nms_thresh) # nms去除重复的限制参数
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names") # 类别，列为list



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile) # 通过cfg配置文件构建Darknet网络
model.load_weights(args.weightsfile) # 加载权重文件
print("Network successfully loaded")

model.net_info["height"] = args.reso # 分辨率，图像高度
inp_dim = int(model.net_info["height"]) # 配置文件中配置的模型高度，模型可以接受的图像高度，也就是长和宽的分辨率
assert inp_dim % 32 == 0 # 确保分辨率是32的倍数
assert inp_dim > 32 # 确保分辨率大于32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda() # 如果系统支持Cuda，将模型放在GPU上


#Set the model in evaluation mode
model.eval()

read_dir = time.time() # 保存读取图片目录文件夹的时间
#Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)] # 拿到image目录下面所有图片的绝对位置
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det): # 输出图片目录，如果没有的话，创建新目录
    os.makedirs(args.det)

load_batch = time.time() # 记录开始读取读片的时间戳
loaded_ims = [cv2.imread(x) for x in imlist] # 存储着图片数量*高*宽*通道数量，pics * HxWxC
# map 函数的第一个参数是调用的函数，这里轮询调用prep_image(loaded_ims[x], inp_dim[x])
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))])) 
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims] # 所有列出照片的（长、高）
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2) # 第一维度不重复，第二维度重复2次，变成了11*4


leftover = 0
if (len(im_dim_list) % batch_size): # 是否图片数量能够被batch size整除
    leftover = 1
if batch_size != 1: # 如果batch_size为1，那么im_batches就不分块了,im_batches为list，内部为1*3*416*416
    num_batches = len(imlist) // batch_size + leftover        # 计算有多少个batch     
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  # 根据batch size对im_batches进行分块
    # im_batches 依然为list，但是len数量就是num_batches了，每一个batch内部的第一个维度不在是1，而是batch_size
write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda() # 将需要检测的数据写入GPU中
    
start_det_loop = time.time() # 记录下所有图片组开始预测的时间戳
for i, batch in enumerate(im_batches): # 加载batch
#load the image 
    start = time.time() # 记录单个batch开始运行的时间戳
    if CUDA:
        batch = batch.cuda() # 放到GPU中
    with torch.no_grad(): # 预测阶段，不更新梯度
        prediction = model(Variable(batch), CUDA) # 第一个参数为forward函数中的x，第二个参数为是否存在CUDA加速
        # 这里的prediction输出: torch.Size([（batch size）, 10647, 85])

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time() # 记录单个batch结束的时间戳

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
    
