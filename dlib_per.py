import dlib
import cv2

# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()

# 图片所在路径
img = cv2.imread(r'C:\Users\lenovo\Desktop\1.jpg')
# img = cv2.resize(img,(64,64))
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# 生成 Dlib 的图像窗口
# win = dlib.image_window()
# win.set_image(img)

# 使用detector检测器来检测图像中的人脸
# dlib.pyramid_up(img)
faces = detector(img, 1)
if len(faces) >= 1:
    print("人脸数 / faces in all：", len(faces))
    for i, d in enumerate(faces):
        print("第", i + 1, "个人脸的矩形框坐标：",
              "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(), '\t', "bottom:", d.bottom())
        img = cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

else:
    print("no face")

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)

# 绘制矩阵轮廓
# win.add_overlay(faces)
# 保持图像
# dlib.hit_enter_to_continue()