# import cv2

# cap = cv2.VideoCapture(2)
# if not cap.isOpened():
#     print("Cannot open camera")
# else:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow("Test", frame)
#         cv2.waitKey(0)
#     cap.release()
#     cv2.destroyAllWindows()


# import cv2

# for i in range(5):
#     cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#     if cap.isOpened():
#         print(f"Camera found at index {i}")
#         cap.release()
#     else:
#         print(f"No camera at index {i}")


import cv2

index = 0

# Try different backends
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

for backend in backends:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        print(f"Camera opened at index {index} with backend {backend}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Test", frame)
            cv2.waitKey(3000)
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        print(f"Failed to open camera at index {index} with backend {backend}")


