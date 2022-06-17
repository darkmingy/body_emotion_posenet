import numpy as np
import cv2
import tflite_runtime.interpreter as tflite # 导入tflite
from PIL import Image, ImageFont, ImageDraw
import math

# # 打开摄像头
# class Opencamera():
#     def __init__(self):
#         self.opsetup_camera()
#
#     def opsetup_camera(self):
#         self.cap = cv2.VideoCapture(0)
#         self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
#         # 创建视频显示线程
#         thL = threading.Thread(target=self.openCameraL)
#         thL.start()
#         thR = threading.Thread(target=self.openCameraR)
#         thR.start()
#
#     def openCameraL(self):
#         while self.cap.isOpened():
#             success, frame = self.cap.read()
#             frame = cv2.resize(frame, (640, 240))
#             left = frame[0:240, 0:320]
#             # RGB转BGR
#             left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
#             img = QImage(left.data, left.shape[1], left.shape[0], QImage.Format_RGB888)
#             self.ui.Opclabel1.setPixmap(QPixmap.fromImage(img))
#
# # 打开摄像头
# class Usecamera():
#     def opsetup_camera(self):
#         self.flag = True
#         self.cap = cv2.VideoCapture(0)
#         self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
#         # 创建视频显示线程
#         self.thL = threading.Thread(target=self.openCameraL)
#         self.thL.start()
#         self.thR = threading.Thread(target=self.openCameraR)
#         self.thR.start()
#         self.key_listener = keyboard.Listener(on_press=self.on_press)
#         self.key_listener.start()
#
#     def EndCapture(self):
#         self.flag = False
#         print("right")


# 显示中文标签
def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 25, encoding="utf-8")
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    # if not isinstance(chinese,unicode):
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, fillColor, font)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

# 计算向量角度
def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle

def get_pos(keypoints):
    # 计算右臂与水平方向的夹角
    keypoints = np.array(keypoints)
    v1 = keypoints[5] - keypoints[6]
    v2 = keypoints[8] - keypoints[6]
    angle_right_arm = get_angle(v1, v2)

    # 计算左臂与水平方向的夹角
    v1 = keypoints[7] - keypoints[5]
    v2 = keypoints[6] - keypoints[5]
    angle_left_arm = get_angle(v1, v2)

    # 计算左肘的夹角
    v1 = keypoints[6] - keypoints[8]
    v2 = keypoints[10] - keypoints[8]
    angle_right_elbow = get_angle(v1, v2)

    # 计算右肘的夹角
    v1 = keypoints[5] - keypoints[7]
    v2 = keypoints[9] - keypoints[7]
    angle_left_elbow = get_angle(v1, v2)

    str_pos = ""
    print(angle_left_elbow)
    print(angle_right_elbow)
    print(angle_left_arm)
    print(angle_right_arm)
    # print(" keypoints[12][0] = ",keypoints[12][0]," keypoints[12][1] = ",keypoints[12][1])
    # print(" keypoints[16][0] = ",keypoints[16][0]," keypoints[16][1] = ",keypoints[16][1])
    # print(" keypoints[11][0] = ",keypoints[11][0]," keypoints[11][1] = ",keypoints[11][1])
    # print(" keypoints[15][0] = ",keypoints[15][0]," keypoints[15][1] = ",keypoints[15][1])
    print(" keypoints[10][0] = ", keypoints[10][0], " keypoints[10][1] = ", keypoints[10][1])
    print(" keypoints[9][0] = ", keypoints[9][0], " keypoints[9][1] = ", keypoints[9][1])
    print(" keypoints[1][0] = ",keypoints[1][0]," keypoints[1][1] = ",keypoints[1][1])
    print(" keypoints[2][0] = ",keypoints[2][0]," keypoints[2][1] = ",keypoints[2][1])
    print("----------------------------\n")
    # 设计动作识别规则
    x9_6 = abs(keypoints[9][0]-keypoints[6][0]);
    y9_6 = abs(keypoints[9][1] - keypoints[6][1]);
    x10_5 = abs(keypoints[10][0] - keypoints[5][0]);
    y10_5 = abs(keypoints[10][1]-keypoints[5][1]);

    x9_1 = abs(keypoints[9][0] - keypoints[1][0]);
    x10_2 = abs(keypoints[10][0] - keypoints[2][0]);
    y9_1 = abs(keypoints[9][1] - keypoints[1][1]);
    y10_2 = abs(keypoints[10][1] - keypoints[2][1]);
    y12_16 = abs(keypoints[12][1] - keypoints[16][1]);
    y11_15 = abs(keypoints[11][1] - keypoints[15][1]);
    # 识别快乐 - 抬双手 - 加油姿势
    # and keypoints[9] > keypoints[1] and keypoints[10] >keypoints[2]
    # print()


    if y12_16 <=30 and y11_15<=30:
        str_pos="蹲下"
    elif abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120 and keypoints[10][1] > keypoints[8][1] and keypoints[9][1] >keypoints[7][1]:
        str_pos = "叉腰"
    elif angle_right_arm > 0 and angle_left_arm > 0 and keypoints[10][1] < keypoints[2][1] and keypoints[9][1] < keypoints[1][1]:
        if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
            str_pos = "捂头"
        else:
            str_pos = "举双手"
    elif angle_left_elbow >= 30 and angle_left_elbow <= 75 and angle_right_elbow >= -65 and angle_right_elbow <=-30 and x9_6<= 40  and y9_6<= 40 and x10_5 <= 40 and y10_5<=40:
        str_pos = "怀抱双臂"
    elif (angle_left_elbow <= -10 and angle_left_arm >=-120 and y9_1 <= 60) or(angle_right_elbow <120 and angle_right_arm >= -180 and y10_2 <= 60)   :
        if y9_1 <= 40 and y10_2 <= 40 and x9_1 <= 40 and x10_2 <= 40:
            str_pos="捂脸"
        # elif y9_1 <= 60 and y10_2 <= 60:
        #     str_pos = "加油"
    elif (angle_right_arm < 0 and angle_left_arm > 0 and y9_1 > 50) or (angle_right_arm > 0 and angle_left_arm < 0 and y10_2 >0):
        str_pos = "抬单手"
    elif angle_right_arm < 0 and angle_left_arm < 0:
        str_pos = "正常"

    '''
    elif (angle_left_elbow <= -10 and angle_left_arm >=-120 and y9_1 <= 60) or(angle_right_elbow <120 and angle_right_arm >= -180 and y10_2 <= 60)   :
        if y9_1 <= 40 and y10_2 <= 40:
            str_pos="捂脸"
        else:
            str_pos = "加油"
    elif abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
        str_pos = "生气"  #叉腰
    elif angle_right_arm > 0 and angle_left_arm > 0 :
        if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
            str_pos = "捂头"  # 三角形
        elif keypoints[9][1] < keypoints[1][1] and keypoints[10][1] < keypoints[2][1]:
            str_pos = "抬双手" # 开心
    elif (angle_right_arm < 0 and angle_left_arm > 0  and y9_1 > 50) or (angle_right_arm > 0 and angle_left_arm < 0 and y10_2 >0):
        str_pos = "抬单手" # 抬左手、右手
    elif angle_left_elbow >= 30 and angle_left_elbow <= 75 and angle_right_elbow >= -65 and angle_right_elbow <=-30 and x9_6<= 40  and y9_6<= 30 and x10_5 <= 40 and y10_5<=35:
        str_pos = "怀抱双臂"

    elif angle_right_arm < 0 and angle_left_arm < 0:
        str_pos = "正常"

    # if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120  :
    #     str_pos = "生气"  #叉腰
    # elif angle_right_arm < 0 and angle_left_arm > 0:
    #     str_pos = "抬左手"
    # elif angle_right_arm > 0 and angle_left_arm < 0:
    #     str_pos = "抬右手"
    # elif angle_right_arm > 0 and angle_left_arm > 0:
    #     str_pos = "抬双手"
    #     if abs(angle_left_elbow) < 120 and abs(angle_right_elbow) < 120:
    #         str_pos = "三角形"
'''
    return str_pos


if __name__ == "__main__":
    # 检测模型
    file_model = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

    interpreter = tflite.Interpreter(model_path=file_model)
    interpreter.allocate_tensors()

    # 获取输入、输出的数据的信息
    input_details = interpreter.get_input_details()
    print('input_details\n', input_details)
    output_details = interpreter.get_output_details()
    print('output_details', output_details)

    # 获取PosNet 要求输入图像的高和宽
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # 初始化帧率计算
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    video = "0531_4.mp4"
    # 打开摄像头
    cap = cv2.VideoCapture(video)
    while True:

        # 获取起始时间 - 计算帧率
        t1 = cv2.getTickCount()

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            break
        # 获取图像帧的尺寸
        imH, imW, _ = np.shape(img)

        # 适当缩放
        img = cv2.resize(img, (360,640))

        # 获取图像帧的尺寸 - 获取宽和高
        imH, imW, _ = np.shape(img)

        # BGR 转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 尺寸缩放适应PosNet 网络输入要求
        img_resized = cv2.resize(img_rgb, (width, height))

        # 维度扩张适应网络输入要求 - 加了第0列进去
        input_data = np.expand_dims(img_resized, axis=0)

        # 尺度缩放 变为 -1~+1   - 对像素值进行正则化变为(-1,1)之间
        input_data = (np.float32(input_data) - 128.0) / 128.0

        # 数据输入网络
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 进行关键点检测
        interpreter.invoke()

        # 获取hotmat - 热度图将图像划分网格，每个网格的得分代表当前关节在此网格点附近的概率；偏移图代表xy两个坐标相对于网格点的偏移情况。
        hotmaps = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects 检测到对象的边界框坐标

        # 获取偏移量(offset)
        offsets = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects 检测到的对象的类索引

        # 获取hotmat的 宽 高 以及关键的数目
        h_output, w_output, n_KeyPoints = np.shape(hotmaps)

        # 存储关键点的具体位置
        keypoints = []
        # 关键点的置信度
        score = 0

        for i in range(n_KeyPoints):
            # 遍历每一张hotmap
            hotmap = hotmaps[:, :, i]

            # 获取最大值 和最大值的位置
            max_index = np.where(hotmap == np.max(hotmap))
            max_val = np.max(hotmap)

            # 获取y，x偏移量 前n_KeyPoints张图是y的偏移 后n_KeyPoints张图是x的偏移
            offset_y = offsets[max_index[0], max_index[1], i]
            offset_x = offsets[max_index[0], max_index[1], i + n_KeyPoints]

            # 计算在posnet输入图像中具体的坐标(在输入图像上的 - 257*257)
            pos_y = max_index[0] / (h_output - 1) * height + offset_y
            pos_x = max_index[1] / (w_output - 1) * width + offset_x

            # 计算在源图像中的坐标 - 映射到视频原图像上的坐标
            pos_y = pos_y / (height - 1) * imH
            pos_x = pos_x / (width - 1) * imW

            # 取整获得keypoints的位置
            keypoints.append([int(round(pos_x[0])), int(round(pos_y[0]))])

            # 利用sigmoid函数计算置每一个点的置信度
            score = score + 1.0 / (1.0 + np.exp(-max_val))

        # 取平均得到最终的置信度
        score = score / n_KeyPoints

        if score > 0.5:
            # 标记关键点 - 输入图片 - 圆心位置 - 半径 - 颜色 - 粗细 - 边界类型 - 小数 位数
            # cv2.circle(img,(80,80),30,(0,0,255),-1)
            flag = 1;
            for point in keypoints:
                if flag %2 == 1:
                    cv2.circle(img, (point[0], point[1]), 3, (0, 255, 255), 3)
                else:
                    cv2.circle(img, (point[0], point[1]), 3, (255, 255, 0), 3)
                flag = flag + 1

            # 画关节连接线 -False指的是不画成闭环线
            # 左臂
            cv2.polylines(img, [np.array([keypoints[5], keypoints[7], keypoints[9]])], False, (0, 255, 0), 3)
            # # 右臂
            cv2.polylines(img, [np.array([keypoints[6], keypoints[8], keypoints[10]])], False, (0, 0, 255), 3)
            # # 左腿
            cv2.polylines(img, [np.array([keypoints[11], keypoints[13], keypoints[15]])], False, (0, 255, 0), 3)
            # # 右腿
            cv2.polylines(img, [np.array([keypoints[12], keypoints[14], keypoints[16]])], False, (0, 255, 255), 3)
            # 身体部分
            cv2.polylines(img, [np.array([keypoints[5], keypoints[6], keypoints[12], keypoints[11], keypoints[5]])],
                          False, (255, 255, 0), 3)

            # 计算位置角
            str_pos = get_pos(keypoints)

        # 显示动作识别结果
        img = paint_chinese_opencv(img, str_pos, (0, 5), (255, 0, 0))

        # 显示帧率
        cv2.putText(img, 'FPS: %.2f score:%.2f' % (frame_rate_calc, score), (imW - 350, imH - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # 显示结果
        cv2.imshow('Pos', img)

        # 计算帧率
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
























