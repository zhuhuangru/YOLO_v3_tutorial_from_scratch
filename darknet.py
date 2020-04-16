from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 



def get_test_input(): # 读取测试图片，下面的转化有需要一些手法
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          # Resize to the input dimension 调整分辨率
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # 按照\n字符分割为多行
    lines = [x for x in lines if len(x) > 0]               # 去除空行
    lines = [x for x in lines if x[0] != '#']              # 去除注释行
    lines = [x.rstrip().lstrip() for x in lines]           # 将每一行左右两侧的空白符(包括\n、\r、\t、' ')删除
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # 配置文件每一个新的层都是以[xxxx]开始
            if len(block) != 0:          # 如果block dic内部存在key-value值，那么这是上一个block留下来的参数
                blocks.append(block)     # 将上一个block的添加到blocks序列中
                block = {}               # 并且清零
            block["type"] = line[1:-1].rstrip() # 是一个新层开始，且block内部为空，那么截取第二个字符到导出第二个字符，比如[convolutional] 转化为 type -> convolutional
        else:                            # 不是新一层开始的配置
            key,value = line.split("=")  # 分割key value
            block[key.rstrip()] = value.lstrip() # 写入block中
    blocks.append(block) # 最后一个block需要在外部再添加一次
    
    return blocks # 返回的是由配置文件转化而成的blocks序列，不包含任何神经网络结构


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]     # 整个网络的配置   
    module_list = nn.ModuleList() # 新建一个ModuleList
    prev_filters = 3 # 第一层层数，输入416*416*3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]): # 遍历blocks序列中从第二个（序号为1）到最后一个
        module = nn.Sequential() # 创建Sequentail
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"): # 卷积层
            #Get the info about the layer
            activation = x["activation"] # 第一步，检查最后输出的激活层，YOLOv3的激活层有两个，一个leaky，一个linear，一共75层卷积，其中72层是leaky，3层YOLO输出层前一层是linear，shortcut的输出层也有linear
            try:
                batch_normalize = int(x["batch_normalize"]) # batch normalize，共72个，75层卷积中除了3个YOLO输出层的前面一层，其他都有BN
                bias = False # 有了BN，bias不需要
            except:
                batch_normalize = 0 # 没有BN，需要bias
                bias = True
        
            filters= int(x["filters"]) # 卷积核心数量
            padding = int(x["pad"]) # 是否pdding
            kernel_size = int(x["size"]) # 卷积核大小，YOLOv3，都是3x3或是1x1
            stride = int(x["stride"]) # 步长
        
            if padding: # YOLOv3的卷积层全部是有padding的
                pad = (kernel_size - 1) // 2 # kernal size 为1那么pad = 0 不添加像素，3的话pad为1，四周都添加1个像素
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv) # 这里的index是指第几个网络层（按照全部107层(0-106)来计算）
        
            #Add the Batch Norm Layer
            if batch_normalize: # 添加BN层，没有BN层，那么bias在上面的卷积上自动添加了
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky": # 添加激活层，Linear没有添加，大胆估计，Linear层应该在yolo输出那边
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"): # 上采样层，13*13*n -> 26*26*n
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest") 
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"): # route层，如果route层后面的数字是一个，那么就是把前面的某个层内的输出接过来，如果后面的数字是两个，那么就是将前面两层的输出连接起来
            x["layers"] = x["layers"].split(',') # ,分割的两层
            #Start  of a route
            start = int(x["layers"][0]) # 第一个层，配置时就是负数
            #end, if there exists one.
            try:
                end = int(x["layers"][1]) # 第二个层，配置时候是网络层数的绝对位置，为正数
            except:
                end = 0 # 如果route后面只有一个数字，那么end=0
            #Positive anotation
            if start > 0: 
                start = start - index # 如果start为+，那么配置时候写的就是绝对位置，就需要减去当前的index，再次变成负数
            if end > 0:
                end = end - index # 道理和尚敏的start - index一样
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route) # 创建一个空的层，暂时什么都不操作
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end] # 在output_filters中取出输出的filters数量，也就是卷积核数量
            else:
                filters= output_filters[index + start] # 没有end时，就是直接连接
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut": # YOLOv3的shortcut，都是从-3层+-1层，且还带了Linear激活层
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",") # 读取mask值，这个mask用来表示哪些anchors对这一层有用
            mask = [int(x) for x in mask] # 取整数，其实配置里面都是整数，这一行多余了，可能是为了防止后面程序运行错误准备
    
            anchors = x["anchors"].split(",") # 读取yolo配置文件中的anchors，yolov3输出的3个大小框anchors是配置在一起的，所以这里需要使用mask进行区分
            anchors = [int(a) for a in anchors] # 同样取整
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)] # 改变为[(a,b),(c,d)] 结构，anchors长度减半
            anchors = [anchors[i] for i in mask] # 取出mask标识的anchors，最小分辨率的是0,1,2,26*26的是3,4,5,13*13的是6,7,8

            '''
            [(116, 90), (156, 198), (373, 326)]
            [(30, 61), (62, 45), (59, 119)]
            [(10, 13), (16, 30), (33, 23)]
            '''
    
            detection = DetectionLayer(anchors) # 创建一个空的检测层
            module.add_module("Detection_{}".format(index), detection) # 保存

        module_list.append(module) # 将之前准备的层添加到ModuleList中
        prev_filters = filters # 这一行是保存本层的输出层数，对于卷积层就是这一层的卷积核数量，给下一个卷积层使用
        output_filters.append(filters) # 同时将output的大小保存，提供给shutcut和route层使用
        
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile) # 解析配置文件，返回blocks序列，共有108个block（第一个net信息在内）
        self.net_info, self.module_list = create_modules(self.blocks) # 根据blocks序列生成net_info(网络信息，也就是yolo.cfg最开始配置的[net]部分)，moudle_list是指107层(0-106)全部的网络层
        
    def forward(self, x, CUDA): # 向前传播函数，这里的x是带batch size的，第一个维度就是batch size
        modules = self.blocks[1:] # 去掉第一个配置层，1-107都是网络层
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):     # 遍历所有曾   
            module_type = (module["type"]) # 拿到type，还是和create_modules一样，有convolutional、upsample、shortcut、route、yolo五种层
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x) # 卷积层和上采样曾，可以直接从原先构建的ModuleList中拿出来
    
            elif module_type == "route": # route层还需要构建
                layers = module["layers"] # 还是从配置文件中拿到layers参数
                layers = [int(a) for a in layers] # 取整数
    
                if (layers[0]) > 0: # 对layers后的第一个参数从绝对位置调整为相对位置
                    layers[0] = layers[0] - i
    
                if len(layers) == 1: # 没有第二个参数，那么就是直接连接过去
                    x = outputs[i + (layers[0])] # 从outputs中取出前几层的输出，由于layers[0] <=0 ，所以这里肯定是之前的层
    
                else: # route是将两层连接起来
                    if (layers[1]) > 0: # 调整位置
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]] # 取出，维度为4，第一为batch，第二为filters，第三第四为像素
                    map2 = outputs[i + layers[1]] # 取出，维度为4，第一为batch，第二为filters，第三第四为像素
                    x = torch.cat((map1, map2), 1) # torch.cat 张量拼接，按照维数1进行操作，filters相互叠在一起
                
    
            elif  module_type == "shortcut": # shortcut层
                from_ = int(module["from"]) # 读取数值，yolov3中都是-3
                x = outputs[i-1] + outputs[i+from_] # 将上一层和-3层相加，这里的相加是指张量大小完全一致的情况下，所有对应位置相加生成新的张量
    
            elif module_type == 'yolo':        # 检测输出层
                anchors = self.module_list[i][0].anchors # 这里的0，解释一下，ModuleList中含有的是Sequential，Sequential的第一个才是DetectionLayer()，所以多了一个[0]
                #Get the input dimensions
                inp_dim = int (self.net_info["height"]) # 输入的高度像素
        
                #Get the number of classes
                num_classes = int (module["classes"]) # 种类数量
        
                #Transform 
                x = x.data # 这句话，很重要，可以去搜一搜，最好替换为.detach()，为了兼容老版本，这里不修改。这里输出的张量大小为：
                '''
                torch.Size([1, 255, 13, 13])
                torch.Size([1, 255, 26, 26])
                torch.Size([1, 255, 52, 52])
                '''
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA) # 转化后就变成下面的大小，输出的预测box位置也根据Anchor Box进行了调整
                '''
                torch.Size([1, 507, 85])   13*13*3
                torch.Size([1, 2028, 85])  26*26*3
                torch.Size([1, 8112, 85])  52*52*3
                '''
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1) # 通过第二维度（index=1）将3个输出的张量拼接起来，后续还要做NMS，输出的张量大小为 torch.Size([1（batch size）, 10647, 85])
        
            outputs[i] = x
        
        return detections


    def load_weights(self, weightfile): # 加载权重文件，并且填入构建的网络中
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


