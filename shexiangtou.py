import numpy as np
import cv2 
import os
import cv2 as cv
import copy
def fun(image):

    boxes = []
    confidences = []
    classIDs = []
    (H,W) = image.shape[0:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)#构造了一个blob图像，对原图像进行了图像的归一化，缩放了尺寸 ，对应训练模型

    net.setInput(blob) #将blob设为输入？？？ 具体作用还不是很清楚
    layerOutputs = net.forward(ln)  #ln此时为输出层名称  ，向前传播  得到检测结果

    for output in layerOutputs:  #对三个输出层 循环
        for detection in output:  #对每个输出层中的每个检测框循环
            scores=detection[5:]  #detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
            classID = np.argmax(scores)#np.argmax反馈最大值的索引
            confidence = scores[classID]
            if confidence >0.1:#过滤掉那些置信度较小的检测结果
                box = detection[0:4] * np.array([W, H, W, H])
                #print(box)
                (centerX, centerY, width, height)= box.astype("int")
                # 边框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新检测出来的框
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs=cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)

    return idxs , boxes , confidences , classIDs


weightsPath='/Users/kuimao/Desktop/image.ball2/module/ball_tiny3_0421.weights'# 模型权重文件
configPath='/Users/kuimao/Desktop/image.ball2/module/yolov3-tiny3-1cls.cfg'# 模型配置文件
labelsPath = "/Users/kuimao/Desktop/image.ball2/module/ball.names"# 模型类别标签文件
#初始化一些参数
LABELS = open(labelsPath).read().strip().split("\n")

#加载 网络配置与训练的权重文件 构建网络
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)  
#读入待检测的图像

# 得到 YOLO需要的输出层
ln = net.getLayerNames()
out = net.getUnconnectedOutLayers()#得到未连接层得序号  [[200] /n [267]  /n [400] ]
x = []
for i in out:   # 1=[200]
    x.append(ln[i-1])    # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
ln=x
# ln  =  ['yolo_82', 'yolo_94', 'yolo_106']  得到 YOLO需要的输出层

 

# 0是代表摄像头编号，只有一个的话默认为0
capture = cv.VideoCapture(0)
while True:
    # 调用摄像机
    ref, frame = capture.read()
    frame_resize = cv2.resize(frame, (416, 234))
    frame1 =copy.deepcopy(frame_resize)
    # 输出图像,第一个为窗口名字
    # cv.imshow('PC Camera', frame)

    idxs , boxes , confidences , classIDs = fun(frame1)

    if len(idxs) > 0 :
        box_seq = idxs.flatten()#[ 2  9  7 10  6  5  4]
        if len(idxs)>0:
            for seq in box_seq:
                (x, y) = (boxes[seq][0], boxes[seq][1])  # 框左上角
                (w, h) = (boxes[seq][2], boxes[seq][3])  # 框宽高
                if classIDs[seq]==0: #根据类别设定框的颜色
                    color = [0,0,255]
                else:
                    color = [0,255,0]
                cv2.rectangle(frame_resize, (x, y), (x + w, y + h), color, 2)  # 画框
                text = "{}: {:.4f}".format(LABELS[classIDs[seq]], confidences[seq])
                cv2.putText(frame_resize, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 写字

    cv2.imshow("Image", frame_resize)

    # 等待5秒显示图像，若过程中按“Esc”（key=27）退出
    c = cv.waitKey(5) & 0xff
    if c == 27:
        # 释放所有窗口
        cv.destroyAllWindows()
        print(frame.shape[0:2])
        break