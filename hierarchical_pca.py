import numpy as np
from sklearn.decomposition import PCA
import os


def two_layer_pca(n_hidden, n_out, input_path='frames.npy'):
    """
    :param n_hidden: number of PCA components in the hidden layer
    :param n_out: number of PCA components in the output layer
    :return:
    """
    data = np.load(input_path)
    frames = data.copy()
    del data

    n_frames = frames.shape[0]  # total number of frames in the video
    height = frames.shape[1]
    width = frames.shape[2]
    # contains corresponding color channel image for all frames
    # in the BGR order since this is the order used in OpenCV
    colors = {'b': frames[:, :, :, 2],
              'g': frames[:, :, :, 1],
              'r': frames[:, :, :, 0]}

    # splitting each frame into 4 tiles and applying PCA to each tile
    # flattening each tile of the frame
    top_left_matrix = {key: colors[key][:, :height//2, :width//2].reshape((299, height//2 * width//2)) for key in colors.keys()}
    top_right_matrix = {key: colors[key][:, :height//2, width//2:].reshape((299, height//2 * width//2)) for key in colors.keys()}
    bottom_left_matrix = {key: colors[key][:, height//2:, :width//2].reshape((299, height//2 * width//2)) for key in colors.keys()}
    bottom_right_matrix = {key: colors[key][:, height//2:, width//2:].reshape((299, height//2 * width//2)) for key in colors.keys()}

    matrices = {
        'top_left': top_left_matrix,
        'top_right': top_right_matrix,
        'bottom_left': bottom_left_matrix,
        'bottom_right': bottom_right_matrix
    }

    # to perform inverse we need to store fitted PCA
    # since we fit PCA to each tile separately, in order to perform inverse projection, we need
    # to store fitted PCA
    # since we fit each color channel separately, we need to store channel-specific PCA
    pca_hidden = {}
    for tile in matrices.keys():
        for color in colors.keys():
            pca_hidden[tile, color] = PCA(n_components=n_hidden)

    # FORWARD PASS

    # contains transformed input tiles
    hidden_layer = {}
    for tile in matrices.keys():
        hidden_layer[tile] = []
        for color in colors.keys():
            hidden_layer[tile].append(pca_hidden[tile, color].fit_transform(matrices[tile][color]))
        hidden_layer[tile] = np.stack(hidden_layer[tile], axis=2)

    # glue transformed tiles together
    hidden_output = np.zeros(shape=(n_frames, 2, 2 * n_hidden, 3), dtype=np.float16)
    hidden_output[:, 0, :n_hidden, :] = hidden_layer['top_left']
    hidden_output[:, 0, n_hidden:, :] = hidden_layer['top_right']
    hidden_output[:, 1, :n_hidden, :] = hidden_layer['bottom_left']
    hidden_output[:, 1, n_hidden:, :] = hidden_layer['bottom_right']
    # flattening hidden_output to apply PCA
    hidden_output = np.reshape(hidden_output, newshape=(n_frames, 2 * 2 * n_hidden, 3))

    # clearing dictionary
    hidden_layer.clear()

    # output layer
    # similarly to hidden layer, we store channel-specific PCA
    pca_out = {color: PCA(n_components=n_out) for color in colors.keys()}
    out = {}
    for i, color in enumerate(colors.keys()):
        out[color] = pca_out[color].fit_transform(hidden_output[:, :, i])

    # BACKWARD PASS (reconstruction)

    # backward pass of output layer
    out_inverse = {}
    for color in colors.keys():
        out_inverse[color] = pca_out[color].inverse_transform(out[color])
        # undo flattening
        # and stacking color layers together
        out_inverse[color] = np.reshape(out_inverse[color], newshape=(n_frames, 2, 2 * n_hidden))

    # stacking color layers together
    out_inverse = np.stack([out_inverse[color] for color in colors.keys()], axis=3)

    # clearing dictionary to free up memory
    pca_out.clear()
    matrices.clear()

    # backward pass of hidden layer
    hidden_inverse = {}
    hidden_inverse['top_left'] = out_inverse[:, 0, :n_hidden, :]
    hidden_inverse['top_right'] = out_inverse[:, 0, n_hidden:, :]
    hidden_inverse['bottom_left'] = out_inverse[:, 1, :n_hidden, :]
    hidden_inverse['bottom_right'] = out_inverse[:, 1, n_hidden:, :]

    matrices_restored = {}
    for tile in hidden_inverse.keys():
        matrices_restored[tile] = {}
        for i, color in enumerate(colors.keys()):
            matrices_restored[tile][color] = pca_hidden[tile, color].inverse_transform(hidden_inverse[tile][:, :, i])
            matrices_restored[tile][color] = np.reshape(matrices_restored[tile][color],
                                                        newshape=(n_frames, height//2, width//2)).astype(np.float16)

    # clearing up dictionary to free up memory
    pca_hidden.clear()

    # gluing output back together and saving as .npy file
    result = np.zeros(shape=(n_frames, height, width, 3), dtype=np.uint8)
    result[:, :height // 2, :width // 2, :] = np.stack(
        [matrices_restored['top_left'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)
    result[:, :height // 2, width // 2:] = np.stack([matrices_restored['top_right'][color] for color in colors.keys()],
                                                    axis=3).astype(np.uint8)

    result[:, height // 2:, :width // 2] = np.stack(
        [matrices_restored['bottom_left'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)
    result[:, height // 2:, width // 2:] = np.stack(
        [matrices_restored['bottom_right'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)

    if input_path == 'frames.npy':
        # saving output in case it was application on original frames
        # otherwise we do not need to save the result since it is a restored PCA projection
        # that will be inverted downstream (in hierarchical PCA function)
        output_path = 'results/hierarchical_pca/two_layered_pca/' + str(n_hidden) + '-' + str(n_out) + '/'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        np.save(os.path.join(output_path, 'result.npy'), result)
    return result


def three_layer_pca(n_hidden, n_hidden_2, n_out, input_path='frames.npy'):
    """
    :return:
    """
    data = np.load(input_path)
    frames = data.copy()
    del data

    n_frames = frames.shape[0]  # total number of frames in the video
    height = frames.shape[1]
    width = frames.shape[2]
    # contains corresponding color channel image for all frames
    # in the BGR order since this is the order used in OpenCV
    colors = {'b': frames[:, :, :, 2],
              'g': frames[:, :, :, 1],
              'r': frames[:, :, :, 0]}

    # splitting each frame into 4 tiles and applying PCA to each tile
    # flattening each tile of the frame
    top_left_matrix = {key: colors[key][:, :height // 2, :width // 2].reshape((n_frames, height // 2 * width // 2)) for key in colors.keys()}
    top_right_matrix = {key: colors[key][:, :height // 2, width // 2:].reshape((n_frames, height // 2 * width // 2)) for key in colors.keys()}
    bottom_left_matrix = {key: colors[key][:, height // 2:, :width // 2].reshape((n_frames, height // 2 * width // 2))
                          for key in colors.keys()}
    bottom_right_matrix = {key: colors[key][:, height // 2:, width // 2:].reshape((n_frames, height // 2 * width // 2))
                           for key in colors.keys()}

    matrices = {
        'top_left': top_left_matrix,
        'top_right': top_right_matrix,
        'bottom_left': bottom_left_matrix,
        'bottom_right': bottom_right_matrix
    }


    # to perform inverse we need to store fitted PCA
    # since we fit PCA to each tile separately, in order to perform inverse projection, we need
    # to store fitted PCA
    # since we fit each color channel separately, we need to store channel-specific PCA
    pca_hidden = {}
    for tile in matrices.keys():
        for color in colors.keys():
            pca_hidden[tile, color] = PCA(n_components=n_hidden)

    # FORWARD PASS

    # contains transformed input tiles
    hidden_layer = {}
    for tile in matrices.keys():
        hidden_layer[tile] = []
        for color in colors.keys():
            hidden_layer[tile].append(pca_hidden[tile, color].fit_transform(matrices[tile][color]))
            # print(f"Explained var: {np.sum(pca_hidden[tile, color].explained_variance_ratio_)}")
        hidden_layer[tile] = np.stack(hidden_layer[tile], axis=2)

    hidden = {
        'upper': np.concatenate([hidden_layer['top_left'], hidden_layer['top_right']], axis=1),
        'lower': np.concatenate([hidden_layer['bottom_left'], hidden_layer['bottom_right']], axis=1)
    }

    # next we apply PCA to the image of upper and lower part separately
    # second hidden layer

    pca_hidden_2 = {}

    hidden_layer_2 = {}
    for loc in hidden.keys():
        hidden_layer_2[loc] = []
        for i, color in enumerate(colors.keys()):
            pca_hidden_2[loc, color] = PCA(n_components=n_hidden_2)
            hidden_layer_2[loc].append(pca_hidden_2[loc, color].fit_transform(hidden[loc][:, :, i]))
            # print(f"Explained var: {np.sum(pca_hidden_2[loc, color].explained_variance_ratio_)}")
        hidden_layer_2[loc] = np.stack(hidden_layer_2[loc], axis=2)

    # gluing together result to pass to output layer
    hidden_layer_2 = np.concatenate([hidden_layer_2[loc] for loc in hidden_layer_2.keys()], axis=1)


    # output layer


    # # similarly to hidden layer, we store channel-specific PCA
    pca_out = {color: PCA(n_components=n_out) for color in colors.keys()}
    out = {}
    for i, color in enumerate(colors.keys()):
        out[color] = pca_out[color].fit_transform(hidden_layer_2[:, :, i])
        # print(f"Explained var: {np.sum(pca_out[color].explained_variance_ratio_)}")


    # BACKWARD PASS (reconstruction)

    # backward pass of output layer
    out_inverse = {}
    for color in colors.keys():
        out_inverse[color] = pca_out[color].inverse_transform(out[color])

    # backward pass of second hidden layer
    hidden_2_inv = {}
    hidden_2_inv['upper'] = {color: out_inverse[color][:, :n_hidden_2] for color in colors.keys()}
    hidden_2_inv['lower'] = {color: out_inverse[color][:, n_hidden_2:] for color in colors.keys()}

    hidden_2_inverse = {}
    for loc in hidden.keys():
        hidden_2_inverse[loc] = []
        for i, color in enumerate(colors.keys()):
            hidden_2_inverse[loc].append(pca_hidden_2[loc, color].inverse_transform(hidden_2_inv[loc][color]))
        hidden_2_inverse[loc] = np.stack(hidden_2_inverse[loc], axis=2)

    # backward pass for first hidden layer
    hidden_inverse = {}
    hidden_inverse['top_left'] = hidden_2_inverse['upper'][:, :n_hidden, :]
    hidden_inverse['top_right'] = hidden_2_inverse['upper'][:, n_hidden:, :]
    hidden_inverse['bottom_left'] = hidden_2_inverse['lower'][:, :n_hidden, :]
    hidden_inverse['bottom_right'] = hidden_2_inverse['lower'][:, n_hidden:, :]


    matrices_restored = {}
    for tile in hidden_inverse.keys():
        matrices_restored[tile] = {}
        for i, color in enumerate(colors.keys()):
            matrices_restored[tile][color] = pca_hidden[tile, color].inverse_transform(hidden_inverse[tile][:, :, i])
            matrices_restored[tile][color] = np.reshape(matrices_restored[tile][color],
                                                        newshape=(n_frames, 240, 320)).astype(np.float16)


    # gluing output back together and saving as .npy file
    result = np.zeros(shape=(n_frames, height, width, 3), dtype=np.uint8)
    result[:, :height // 2, :width // 2, :] = np.stack(
        [matrices_restored['top_left'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)
    result[:, :height // 2, width // 2:] = np.stack([matrices_restored['top_right'][color] for color in colors.keys()],
                                                    axis=3).astype(np.uint8)

    result[:, height // 2:, :width // 2] = np.stack(
        [matrices_restored['bottom_left'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)
    result[:, height // 2:, width // 2:] = np.stack(
        [matrices_restored['bottom_right'][color] for color in colors.keys()],
        axis=3).astype(np.uint8)

    # saving npy output
    output_path = 'results/hierarchical_pca/three_layered_pca/' +str(n_hidden) + '-' + str(n_hidden_2) + '-' + str(n_out) + '/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    np.save(os.path.join(output_path, 'result.npy'), result)


def hierarchical_pca():
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
    pca_init ={ color: PCA(n_components=64) for color in colors.keys()}

    pca_result = {}
    for color in colors.keys():
        pca_result[color] = np.reshape(pca_init[color].fit_transform(colors_matrix[color]), newshape=(299, 8, 8))
    np.save('pca.npy', np.stack([pca_result[color] for color in colors.keys()], axis=3))

    result = two_layer_pca(4, 1, input_path='pca.npy')
    result = np.reshape(result, newshape=(299, 64, 3))
    out = []
    for i, color in enumerate(colors.keys()):
        out.append(np.reshape(pca_init[color].inverse_transform(result[:, :, i]), newshape=(299, 480, 640)))
    np.save('hierarchical_pca.npy', np.stack(out, axis=3))



if __name__ == "__main__":
    # n_hidden = 16
    # n_hidden_2 = 4
    # n_out = 1
    # # # two_layer_pca(n_hidden, n_out)
    # three_layer_pca(n_hidden, n_hidden_2, n_out)
    hierarchical_pca()


