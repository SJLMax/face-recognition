import cv2


# 图片名
filename =r'C:\Users\lenovo\Desktop\3.jpg'


def detect(filename):

    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 读取图片
    img = cv2.imread(filename)
    # img = cv2.resize(img, (64,64))
    # 转灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    # scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
    # minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
    #         如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
    #         如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
    #         这种设定值一般用在用户自定义对检测结果的组合程序上；
    faces = face_cascade.detectMultiScale(img, 1.1, 3)
    print(faces)
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 命名显示窗口
    cv2.namedWindow('people')
    # 显示图片
    cv2.imshow('people',img)
    # 保存图片
    # cv2.imwrite('./1.jpg', img2)
    # 设置显示时间,0表示一直显示
    cv2.waitKey(0)


detect(filename)