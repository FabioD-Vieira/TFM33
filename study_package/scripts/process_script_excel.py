import cv2
import numpy as np

from xlwt import Workbook

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

process = Process(lut, pool_dim)

cap = cv2.VideoCapture("../../images/videos/vid07.h264")

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0, 0, "x")
sheet1.write(0, 1, "y")
sheet1.write(0, 2, "angle")

index = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    pos, angle = process.start(frame)

    if pos is not None:

        sheet1.write(index, 0, pos[0])
        sheet1.write(index, 1, pos[1])
        sheet1.write(index, 2, angle)

        index += 1

wb.save('vid07.xls')

cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
