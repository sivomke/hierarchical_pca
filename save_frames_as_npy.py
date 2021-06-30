import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    cap = cv2.VideoCapture('traffic.avi')
    ret, frame = cap.read()
    i = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if frame is not None:
            i += 1
            frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame_rgb)
            cv2.imshow('ImageWindow', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    print(i)
    frames = np.array(frames)
    print(frames.shape)
    cap.release()
    cv2.destroyAllWindows()

    np.save('frames.npy', frames)


if __name__ == "__main__":
    main()