import mediapipe as mp
import argparse
import torch
import numpy as np
import time
import os
import cv2
import torch.nn.functional as F
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ControlBilibili import ControlBilibili
from flask import Flask # 导入必要的模块
from flask import render_template,Response
app = Flask(__name__)  # 创建 Flask 应用程序实例
# 路由函数，处理来自用户的请求，返回一个响应
@app.route('/')
def index():
    image_folder = os.path.join('static','images')
    images = os.listdir(image_folder)
    return render_template("index.html", images=images)
    #return render_template('index.html') # 渲染名为 'index.html' 的模板并返回它
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils
handLmsStyle=mpDraw.DrawingSpec(color=(0,0,255))#改变点的颜色
handConStyle=mpDraw.DrawingSpec(color=(0,0,0))#改变绘制手部连接线的颜色
def PositonDetection(img_):
    #输入图片预处理
    imgHeight = img_.shape[0]  # 图片高度
    imgWidth = img_.shape[1]  # 图片宽度
    img_ = cv2.resize(img_, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
    img_ = img_.astype(np.float32)
    img_ = (img_ - 128.) / 256.
    img_ = img_.transpose(2, 0, 1)
    img_ = torch.from_numpy(img_)
    img_ = img_.unsqueeze_(0)
    if use_cuda:
        img_ = img_.cuda()  # (bs, 3, h, w)
    pre_ = model_(img_.float())
    outputs = F.softmax(pre_, dim=1)
    outputs = outputs[0]
    output = outputs.cpu().detach().numpy()
    output = np.array(output)
    max_index = np.argmax(output)
    print('检测结果：', max_index)
    return max_index
def rectanglepos(xleft,xright,yleft,yright,imgHeight,imgWidth):
    if xleft - 30<0:
        xleft=0
    if yleft - 30<0:
        yleft=0
    if xright + 30>imgWidth:
        xright=imgWidth
    if yright + 30>imgHeight:
        yright=imgHeight
    return xleft,xright,yleft,yright
def classify(img):
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把图片BGR转换成RGB
                result = hands.process(imgRGB)  # 检测手部
                imgHeight = img.shape[0]  # 图片高度
                imgWidth = img.shape[1]  # 图片宽度
                xPos = []
                yPos = []
                img_ = img.copy()
                if result.multi_hand_landmarks:  # 如果检测到手，画出检测到的所有手的标记点，并用线连接
                    for handLms in result.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                        #mpDraw.draw_landmarks(img_, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                        for i, lm in enumerate(handLms.landmark):  # 每个点的坐标
                            xPos.append(int(lm.x * imgWidth))
                            yPos.append(int(lm.y * imgHeight))  # 转换坐标格式
                    xleft = np.min(xPos)
                    xright = np.max(xPos)
                    yleft = np.min(yPos)
                    yright = np.max(yPos)
                    xleft, xright, yleft, yright=rectanglepos(xleft, xright, yleft, yright, imgHeight, imgWidth)
                    img_ = img_[yleft:yright, xleft:xright]  # 仅把检测到的手送入进行分类
                    max_index = PositonDetection(img_)
                    return max_index
def gen_img():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        else:
            #将帧转换为 JPEG 格式
            max_index=classify(img)
            if max_index == 0:
                ControlBilibili.speedDown()
            elif max_index == 1:
                ControlBilibili.stopAndPlayVideo()
            elif max_index == 2:
                ControlBilibili.mute()
            elif max_index == 3:
                ControlBilibili.speedUp()
            else:
                continue
            ret, buffer = cv2.imencode('.jpg', img)
            img_copy = buffer.tobytes()
        # 使用 Flask 的 Response 对象将图像帧传递给 HTML 页面
            yield (b'--img\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img_copy + b'\r\n')
@app.route('/video_feed')
def video_feed():
    # 将嵌入式视频传递给 HTML 页面
    return Response(gen_img(),
                    mimetype='multipart/x-mixed-replace; boundary=img')
if __name__ == "__main__":  # 如果脚本是从命令行启动，运行 Flask 应用程序
    # ---------------------------------------------------------------------------
    GPUS = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
    # ---------------------------------------------------------------- 构建模型
    num_classes = 4
    img_size = (192, 192)
    test_model = './model_exp_cla4/2022-10-29_21-48-24/resnet_18-size-256_epoch-130.pth'
    model_ = resnet18(num_classes=num_classes, img_size=img_size[0])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()  # 设置为前向推断模式
    # 加载测试模型
    if os.access(test_model, os.F_OK):
        chkpt = torch.load(test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(test_model))
    app.run(host='127.0.0.1', port=8001,debug=True)