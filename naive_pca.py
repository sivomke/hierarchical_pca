import numpy as np
from sklearn.decomposition import PCA


def naive_pca(n_components):
    """
    :param n_components: number of PC used for reconstruction
    :return:
    """
    data = np.load('frames.npy')
    frames = data.copy()
    del data
    # contains corresponding color channel image for all frames
    colors = {'r': frames[:, :, :, 0],
              'g': frames[:, :, :, 1],
              'b': frames[:, :, :, 2]}
    colors_matrix = {
        'r': colors['r'].reshape((299, 480 * 640)),
        'g': colors['g'].reshape((299, 480 * 640)),
        'b': colors['b'].reshape((299, 480 * 640))
    }

    n_features = 480 * 640
    pca = PCA(n_components=n_components)
    pca_result = {}
    pca_result['r'] = np.reshape(pca.inverse_transform(pca.fit_transform(colors_matrix['r'])), newshape=(299, 480, 640))
    pca_result['g'] = np.reshape(pca.inverse_transform(pca.fit_transform(colors_matrix['g'])), newshape=(299, 480, 640))
    pca_result['b'] = np.reshape(pca.inverse_transform(pca.fit_transform(colors_matrix['b'])), newshape=(299, 480, 640))
    # result = np.stack([pca_result['b'], pca_result['g'], pca_result['r']], axis=3).astype(np.uint8) # stacking as BGR for OpenCV
    # np.save('results/naive_pca/result.npy', result)


if __name__ == "__main__":
    naive_pca(1)

