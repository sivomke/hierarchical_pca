import numpy as np
import cv2

# output path for absolute diff between original video and PCA-reconstruction
OUTPUT_DIFF_PATH = 'results/naive_pca/abs_diff.avi'
# output path for PCA reconstruction video (with 1 PC)
OUTPUT_PCA_PATH = 'results/naive_pca/inverse_pca.avi'


def save_abs_diff(path_to_video, path_to_reconstruction, output_path):
    original_video = cv2.VideoCapture(path_to_video)
    inverse_pca_video = cv2.VideoCapture(path_to_reconstruction)

    # calculating params for output video
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(original_video.get(cv2.CAP_PROP_FPS))  # frames per second

    # opening an output video for abs diff between original and reconstructed frames
    abs_diff_video = cv2.VideoWriter(output_path, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                                   frameSize=(width, height))

    while True:
        ret, frame = original_video.read()
        ret, frame_pca = inverse_pca_video.read()

        if frame is not None and frame_pca is not None:
            abs_diff = cv2.absdiff(frame, frame_pca)
            # we will threshold the obtained absolute difference value to reduce the number of artifacts
            # and separate foreground from background
            threshold = 50
            bg_color = (0, 0, 0)
            _, mask = cv2.threshold(src=cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY), thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)
            mask = np.array(mask)
            abs_diff[mask == 0] = bg_color

            cv2.imshow('frame', abs_diff)
            abs_diff_video.write(abs_diff)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    # closing input and output videos
    original_video.release()
    inverse_pca_video.release()
    abs_diff_video.release()
    cv2.destroyAllWindows()


def abs_diff(input_path, output_path):
    """
    :param input_path: path to extracted background
    :param output_path: path to folder where to save absolute difference video
    :return:
    """
    data = np.load(input_path)
    cap = cv2.VideoCapture('traffic.avi')

    # calculating params for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # frames per second
    # opening an output video for abs diff between original and reconstructed frames
    abs_diff_video = cv2.VideoWriter(output_path, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                                   frameSize=(width, height))
    # # opening an output video for reconstructed frames
    # inverse_pca_video = cv2.VideoWriter(OUTPUT_PCA_PATH, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
    #                                  frameSize=(width, height))
    # opening an output video for reconstructed frames
    for i in range(data.shape[0]):
        ret, frame = cap.read()
        frame_pca = np.array(data[i, :, :, :], dtype=np.uint8)
        abs_diff = cv2.absdiff(frame, frame_pca)
        # threshold = 50
        # bg_color = (0, 0, 0)
        # _, mask = cv2.threshold(src=cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY), thresh=threshold, maxval=1,
        #                         type=cv2.THRESH_BINARY)
        # mask = np.array(mask)
        # abs_diff[mask == 0] = bg_color

        # inverse_pca_video.write(frame_pca)
        abs_diff_video.write(abs_diff)
        cv2.imshow('frame', abs_diff)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # closing input and output videos
    cap.release()
    # inverse_pca_video.release()
    abs_diff_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # input_path = 'results/hierarchical_pca/three_layered_pca/1-1-1/result.npy'
    # output_path = 'results/hierarchical_pca/three_layered_pca/1-1-1/abs_diff.avi'

    input_path = 'hierarchical_pca.npy'
    output_path = 'hierarchical_pca_abs_diff.avi'
    abs_diff(input_path, output_path)
    # # path to original video
    # path_to_video = 'traffic.avi'
    # path_to_reconstruction = 'results/pca_on_tiles/result.avi'
    # output_path = 'results/pca_on_tiles/abs_diff.avi'
    # save_abs_diff(path_to_video, path_to_reconstruction, output_path)