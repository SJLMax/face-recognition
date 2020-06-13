# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib
import os
import shutil
import time

# 实例化 detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray


def face_detect(sourcePath, targetPath, invalidPath, invalidPath_m,*suffix):
    try:
        ImagePaths = getAllPath(sourcePath, *suffix)
        count = 1
        for imagePath in ImagePaths:
            img = cv2.imread(imagePath)
            if type(img) != str:
                faces = detector(img, 1)  # 参数1表示我们对图像进行向上采样1倍，这将使一切变的更大进而让我们检测出更多的人脸
                # print("face：number:", len(faces))
                if len(faces)==1:
                    for i, d in enumerate(faces):
                        print("第", i + 1, "个人脸的矩形框坐标：",
                              "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(), '\t', "bottom:",
                              d.bottom())
                        # cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                        # 以时间戳和读取的排序作为文件名称
                        listStr = [str(int(time.time())), str(count)]
                        fileName = ''.join(listStr)
                        l=d.left()
                        t=d.top()
                        r=d.right()
                        b=d.bottom()
                        if l<0:
                            l=0
                        if t<0:
                            t=0
                        X = int(l * 0.3)
                        W = min(int((r) * 1.3), img.shape[1])
                        Y = int(t * 0.3)
                        H = min(int((b) * 1.3), img.shape[0])
                        f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                        cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                        count += 1
                        print(imagePath + " have face")
                elif len(faces)>1:
                    shutil.move(imagePath, invalidPath_m)
                else:
                    #print()
                    shutil.move(imagePath, invalidPath)
    except IOError:
        print("Error")

    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)


def face_landmark():
    img = cv2.imread(r'C:\Users\lenovo\Desktop\2.JPG')
    # 人脸数rects
    rects = detector(img, 1)
    print(len(rects))
    print(rects)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        print(predictor(img, rects[i]).parts())
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            print(idx, pos)

            # 利用cv2.circle给每个特征点画一个圈，共68个
            imr = cv2.circle(img, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(r'C:\Users\lenovo\Desktop\2-1.JPG',img)
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    # targetPath = r'C:\Users\lenovo\D   # invalidPath = r'C:\Users\lenovo\Desktop\tensor+CNN_multi\CNN\test_pic\noface'
    # invalidPath = r'C:\Users\lenovo\Desktop\image_all\0_noface'
    # invalidPath_m = r'C:\Users\lenovo\Desktop\image_all\0_facemore'
    # sourcePath = r'C:\Users\lenovo\Desktop\tensor+CNN_multi\CNN\test_pic\2'
    # #sourcePath = r'C:\Users\lenovo\Desktop\image_all\0'
    # targetPath = r'C:\Users\lenovo\Desktop\tensor+CNN_multi\CNN\test_pic\faceOfPeaple'esktop\image_all\0_face'

    # face_detect(sourcePath, targetPath, invalidPath, invalidPath_m, '.jpg', '.JPG', 'png', 'PNG')
    face_landmark()



