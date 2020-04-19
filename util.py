from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2): # 计算两个框的交并比
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True): # 将模型输出结果prediction, 转化为需要的格式

    
    batch_size = prediction.size(0) # 拿到第一维的batch size，这里根据命令行输入的配置进行调整
    stride =  inp_dim // prediction.size(2) # 第三维数据，13、26、52，计算步长
    grid_size = inp_dim // stride # 再反过来计算一次，其实就是13、26、52
    bbox_attrs = 5 + num_classes # 类别数量 + 5 就是每个Anchor Box输出的张量深度
    num_anchors = len(anchors) # 3
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size) # 改变输入的张量形状，后面的参数是变化后的形状 torch.Size([1, 255, 169]) torch.Size([1, 255, 676]) torch.Size([1, 255, 2704])
    prediction = prediction.transpose(1,2).contiguous() # 将第2维和第3维交换，然后contiguous返回深度拷贝的张量， torch.Size([1, 169, 255]) torch.Size([1, 676, 255]) torch.Size([1, 2704, 255])
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs) # 再来一次转换，torch.Size([1, 507, 85]) torch.Size([1, 2028, 85]) torch.Size([1, 8112, 85])
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors] # 将anchor 像素级的参数转化为对单个cell的大小参数

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # 最后长度85位置的第一位为预测的中心点x，输出范围在0-1之间
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) # 最后维度85位置的第二位为预测的中心点y，输出范围在0-1之间
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) # 第五位为object置信度
    
    #Add the center offsets， 下面这一段是在输出的矩阵上添加偏移
    grid = np.arange(grid_size) # np.arange 产生一个0-grid_size-1的np数组
    a,b = np.meshgrid(grid, grid) # 生成2个2维数组，13*13

    x_offset = torch.FloatTensor(a).view(-1,1) # 这里的view相当于reshape，第一个参数-1表示不知道多少维度，第二个参数1表示reshape到1行 torch.Size([169, 1])
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda() # torch.Size([169, 1])
        y_offset = y_offset.cuda() # torch.Size([169, 1])

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0) # 第一步cat转化为169*2, repeat复制，view(-1,2)第二维度修改为2，torch.Size([507, 2])，unsqueeze增加维度，torch.Size([1, 507, 2])

    prediction[:,:,:2] += x_y_offset # 0，1两个位置加上x_y_offset，这个是矩阵操作，prediction和x_y_offset的维度是一样的；

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors) # 格式应该i还是[(,),(,),(,)]

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors # 2，3位置
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction # 返回修正后的数据

def write_results(prediction, confidence, num_classes, nms_conf = 0.4): # 这个函数接上面的模型输出
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) # 从batch*outputchannel*85取出batch*outputchannel*4/85与置信度判断，大于置信度的为1，小于为0. 转化为float之后，再添加一个维度
    # conf_mask 大小应该是batch * output_channel
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim): # resize图片大小
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0] # 获得当前图片的宽度和高度
    w, h = inp_dim                            # 网络需要输入图片的高和宽
    new_w = int(img_w * min(w/img_w, h/img_h))# 找到缩放比例最小的这个，然后按照这个比例进行缩放，就是说如果长比宽大，那么缩放的长就是416，宽另行计算
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC) # 将图片进行缩放，调整到某一个边是416长度
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128) # 产生一个416*416*3的矩阵，内部填充128

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image # 将缩放后的图片放入这个np中，上下居中，左右居中；如果是训练的话，对应训练的label也需要进行变换；同时网络预测结果输出后，需要返回到原图中；
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim))) # 这里输出一个inp_dim * inp_dim 大小的图像，图像等比例缩放，两边添加128中间值
    img = img[:,:,::-1].transpose((2,0,1)).copy() # 转置,将 416 * 416 * 3 转化为 3 * 416 * 416
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # 图片所有数值/255，在第0维度加上一个维度，成为1*3*416*416
    return img

def load_classes(namesfile): # 加载检测的类别，使用\n分开，并且去掉最后一个空白行
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
