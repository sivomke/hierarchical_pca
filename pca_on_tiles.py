import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


def pca_on_tiles(n_components):
    """
    :param n_components: number of PCA components used for reconstruction of frames from each tile
    :return:
    """
    data = np.load('frames.npy')
    frames = data.copy()
    del data

    print(frames[:, :, :, 0].shape)
    # contains corresponding color channel image for all frames
    # in the BGR order since this is the order used in OpenCV
    colors = {'b': frames[:, :, :, 2],
              'g': frames[:, :, :, 1],
              'r': frames[:, :, :, 0]}

    # splitting each frame into 4 tiles and applying PCA to each tile
    # flattening each tile of the frame
    top_left_matrix = {key: colors[key][:, :240, :320].reshape((299, 240 * 320)) for key in colors.keys()}
    top_right_matrix = {key: colors[key][:, :240, 320:].reshape((299, 240 * 320)) for key in colors.keys()}
    bottom_left_matrix = {key: colors[key][:, 240:, :320].reshape((299, 240 * 320)) for key in colors.keys()}
    bottom_right_matrix = {key: colors[key][:, 240:, 320:].reshape((299, 240 * 320)) for key in colors.keys()}


    matrices = {
        'top_left': top_left_matrix,
        'top_right': top_right_matrix,
        'bottom_left': bottom_left_matrix,
        'bottom_right': bottom_right_matrix
    }

    pca = PCA(n_components=n_components)
    for tile in matrices.keys():
        # for simplicity for each color channel we use the same number of principle components
        result = []
        print(tile)
        avg = 0
        for key in colors.keys():
            result.append(np.reshape(pca.inverse_transform(pca.fit_transform(matrices[tile][key])),
                                              newshape=(299, 240, 320)))
            avg += np.sum(pca.explained_variance_ratio_)
        print(avg/3)
        result = np.stack(result, axis=3).astype(np.uint8)  # stacking as BGR for OpenCV
        np.save('results/pca_on_tiles/' + str(tile) + '_result.npy', result)


if __name__ == "__main__":
    pca_on_tiles(1)