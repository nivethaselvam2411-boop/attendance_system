import cv2

cap = cv2.VideoCapture("http://10.156.115.152:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    cv2.imshow("Mobile Camera Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
