#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : opencv_dnn.py
# @Author: Shang
# @Date  : 2020/6/13


from __future__ import division
import cv2


def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return bboxes


if __name__ == '__main__':
    # 加载人脸检测器
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    conf_threshold = 0.7
    print('Loading...')
    # 输入图片

    path = r'C:\Users\lenovo\Desktop\3.jpg'
    img = cv2.imread(path)
    # img = cv2.resize(img, (200, 200))
    bboxes= detectFaceOpenCVDnn(net, img)
    frameHeight = img.shape[0]
    if len(bboxes)==0:
        output=[]
        print('抱歉，未检测到人脸')
    else:
        for i in bboxes:
            img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (171, 207, 49), int(round(frameHeight / 120)), 8)
            # img = cv2.putText(img, str(j), (i[0], i[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 142, 255), 1, cv2.LINE_AA)
    print(bboxes)
    cv2.imshow("Face Detection Comparison", img)
    cv2.waitKey(0)






