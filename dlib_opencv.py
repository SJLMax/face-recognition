import numpy as np
import tensorflow as tf
import dlib
import cv2

# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 使用 opencv 人脸检测器 haarcascade_frontalface_alt.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 类别信息
class_dict = {0: 'man', 1: 'woman'}

# 最新模型地址
model_path = '/home/ugrad/Shang/CNN/model/model_new/'
meta_graph = 'model-49-2020-03-11 13:20:20.meta'

# 测试图片
img_path = r'C:\Users\lenovo\Desktop\image_all\img2.JPG'


def face_detect(path):
    print('Loading...')
    img = cv2.imread(path)
    # img = cv2.resize(img, (64, 64))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用detector检测器来检测图像中的人脸
    faces = detector(img, 1)
    print('------调用Dlib人脸检测分类器------')
    print('人脸数：', len(faces))
    if len(faces) > 1:
        for i, d in enumerate(faces):
            img = cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 0, 0), 2)
        print('检测结果为多人哦！请确定您要性别识别的对象。')
    elif len(faces) == 1:
        for i, d in enumerate(faces):
            l = d.left()
            t = d.top()
            r = d.right()
            b = d.bottom()
            if l < 0:
                l = 0
            if t < 0:
                t = 0
            X = int(l * 0.3)
            W = min(int(r * 1.3), img.shape[1])
            Y = int(t * 0.3)
            H = min(int(b * 1.3), img.shape[0])
            img2 = cv2.resize(img[Y:H, X:W], (W - X, H - Y))  # 裁剪后人脸
            img_final = np.asarray(img2/255, np.float32)
            data = []
            data.append(img_final)
            gender_recognition(data, model_path, meta_graph)  # Dlib后性别分类
            img = cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 0, 0), 2)
    else:
        print('------继续调用OpenCV人脸检测分类器------')
        img = face_detector_opencv(img)
    print('------(如有误差，敬请谅解，还在更新中^^)------')
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def face_detector_opencv(img_cv):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_cv = face_cascade.detectMultiScale(img_cv, 1.1, 3)
    print('人脸数：', len(faces_cv))
    if len(faces_cv) > 1:
        for (x, y, w, h) in faces_cv:
            img_cv = cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print('检测结果为多人哦！请输入明确的性别识别对象。')
    elif len(faces_cv) == 1:
        # 绘制人脸矩形框
        for (x, y, w, h) in faces_cv:
            X = int(x * 0.3)
            W = min(int((x + w) * 1.3), img_cv.shape[1])
            Y = int(y * 0.3)
            H = min(int((y + h) * 1.3), img_cv.shape[0])
            img2 = cv2.resize(img_cv[Y:H, X:W], (W - X, H - Y))
            img_final = np.asarray(img2 / 255, np.float32)
            data = []
            data.append(img_final)
            gender_recognition(data, model_path, meta_graph)  # opencv后性别分类
            img_cv = cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        print('抱歉,未检测到人脸!')
    return img_cv



def gender_recognition(data, model_path, meta_graph):
    tf.reset_default_graph()  # 清除过往tensorflow数据记录
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + meta_graph)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        # sess：表示当前会话，之前保存的结果将被加载入这个会话
        # 设置每次预测的个数
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}
        logits = graph.get_tensor_by_name("logits_eval:0")  # eval功能等同于sess(run)
        classification_result = sess.run(logits, feed_dict)
        # 打印出预测矩阵
        # print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        # print(tf.argmax(classification_result, 1).eval())
        # print("label:", label)
        # 根据索引通过字典对应分
        output = tf.argmax(classification_result, 1).eval()
        for i in range(len(output)):
            print("性别预测为:" + class_dict[output[i]])


if __name__ == '__main__':
    face_detect(img_path)




