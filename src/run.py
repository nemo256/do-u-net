import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import glob
import os

import model, data

import tkinter
matplotlib.use('TkAgg')


def run_training(model_name):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    do_unet = model.DO_UNet(train_img_files,
                            test_img_files)

    if not os.path.exists(f"models/Test_scale_best.h5"):
        do_unet.fit(model_name,
                    epochs=1,
                    imgs_per_epoch=64,
                    batchsize=2,
                    workers=2)

    imgs, mask, edge = data.load_data(test_img_files)
    img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask, edge=edge, padding=200, input_size=380, output_size=196)

    ind = 200
    pred = np.array([img_chips[ind]])
    pred.shape
    print(pred)

    # output = do_unet.predict(pred)
    # output = np.squeeze(output)
    # output.shape

    # fig = plt.figure(figsize=(15, 10), dpi=80)
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # ax = fig.add_subplot(2, 3, 1)
    # ax.imshow(img_chips[ind])
    # ax = fig.add_subplot(2, 3, 2)
    # ax.imshow(edge_chips[ind])
    # ax = fig.add_subplot(2, 3, 3)
    # ax.imshow(mask_chips[ind])
    # ax = fig.add_subplot(2, 3, 5)
    # ax.imshow(output[0])
    # ax = fig.add_subplot(2, 3, 6)
    # ax.imshow(output[1])


if __name__ == '__main__':
    run_training('Test_scale')
