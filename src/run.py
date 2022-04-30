import numpy as np
import matplotlib.pyplot as plt

import glob
import os

import model, data


def run_training(model_name):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    do_unet = model.get_do_unet()

    # Load best weights if they exist (checkpoint)
    if os.path.exists(f"models/Test_scale_best.h5"):
        do_unet.load_weights(f"models/Test_scale_best.h5")
    else:
        # If not, train anew
        do_unet.fit(model_name,
                    epochs=100,
                    imgs_per_epoch=1000,
                    batchsize=8,
                    workers=8)

    imgs, mask, edge = data.load_data(test_img_files)
    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask, edge=edge, padding=200, input_size=188, output_size=196)

    index = 3

    image = np.array([img_chips[index]])

    output = do_unet.predict(image)
    output = np.squeeze(output)

    fig = plt.figure(figsize=(25, 12), dpi=80)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(img_chips[index])
    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(edge_chips[index])
    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(mask_chips[index])
    ax = fig.add_subplot(2, 4, 4)
    ax.imshow((mask_chips[index] - edge_chips[index]) > 0)
    ax = fig.add_subplot(2, 4, 6)
    ax.imshow(output[0])
    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(output[1])
    ax = fig.add_subplot(2, 4, 8)
    ax.imshow((output[1] - output[0]) > 0)

    plt.savefig("sample.png")


if __name__ == '__main__':
    run_training('Test_scale')
