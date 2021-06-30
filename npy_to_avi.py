import numpy as np
import cv2


def npy_to_avi(input_path, output_path):
    data = np.load(input_path)
    # calculating params for output video
    width = int(data.shape[2])
    height = int(data.shape[1])
    fps = 30  # frames per second

    output_video = cv2.VideoWriter(output_path, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                                     frameSize=(width, height))
    for i in range(data.shape[0]):
        frame = np.array(data[i, :, :, :], dtype=np.uint8)
        output_video.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # output video
    output_video.release()
    cv2.destroyAllWindows()


# combining tiles back together and saving result video
def save_glued():
    tile_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    tiles = {}
    for name in tile_names:
        input_path = 'results/pca_on_tiles/' + name + '_result.npy'
        tiles[name] = np.load(input_path)

    num_frames = tiles['top_left'].shape[0]

    # calculating params for output video
    width = int(tiles['top_left'].shape[2])
    height = int(tiles['top_left'].shape[1])
    fps = 30  # frames per second

    result = np.zeros(shape=(tiles['top_left'].shape[0], height * 2, width * 2, 3))

    result[:, :height, :width] = tiles['top_left']
    result[:, :height, width:] = tiles['top_right']
    result[:, height:, :width] = tiles['bottom_left']
    result[:, height:, width:] = tiles['bottom_right']

    np.save('results/pca_on_tiles/result.npy', result)

    input_path = 'results/pca_on_tiles/result.npy'
    output_path = 'results/pca_on_tiles/result.avi'

    npy_to_avi(input_path, output_path)

    for key in tile_names:
        del tiles[key]



if __name__ == "__main__":
    # converting .npy to .avi
    # tile_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    # for name in tile_names:
    #     output_path = 'results/pca_on_tiles/' + name + '_result.avi'
    #     input_path = 'results/pca_on_tiles/' + name + '_result.npy'
    #     npy_to_avi(input_path, output_path)

    # save_glued()
    # input_path = 'results/pca_on_tiles/result.npy'
    # output_path = 'results/pca_on_tiles/result.avi'
    input_path = 'hierarchical_pca.npy'
    output_path = 'hierarchical_pca.avi'
    npy_to_avi(input_path, output_path)