import cv2

cap1 = cv2.VideoCapture(1)  # right
cap2 = cv2.VideoCapture(2)  # left

num = 1

while cap1.isOpened() and cap1.isOpened():
    succes1, img1 = cap1.read()  # right
    succes2, img2 = cap2.read()  # left

    k = cv2.waitKey(5)

    if k == 27:  # press Esc
        break

    elif k == ord("s"):  # press s
        cv2.imwrite("images\img_left\Left" + str(num) + ".png", img2)
        cv2.imwrite("images\img_right\Right" + str(num) + ".png", img1)
        print("No.{num} saved!".format(num=num))
        num += 1

    cv2.imshow("right", img1)
    cv2.imshow("left", img2)


cap1.release()
cap2.release()
cv2.destroyAllWindows()


# name = input()
# print(name)
