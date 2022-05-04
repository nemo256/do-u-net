import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
import model, data


# train the model
def train(model_name):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    do_unet = model.DO_UNet(train_img_files,
                            test_img_files)

    do_unet.fit(model_name,
                epochs=100,
                imgs_per_epoch=1000,
                batchsize=8,
                workers=8)


# extract number of image chips for an image
def get_sizes(img,
              offset=150,
              input=188,
              output=100):
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


# reshape numpy arrays
def reshape(img,
            size_x,
            size_y,
            type='input'):
    if type == 'input':
        return img.reshape(size_x, size_y, 188, 188, 1)
    elif type == 'output':
        return img.reshape(size_x, size_y, 100, 100, 1)
    else:
        print(f'Invalid type: {type} (input, output)')


# concatenate images
def concat(imgs):
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:,:]])


# predict (segment) image and save a sample output
def predict(image = glob.glob('data/test/Im037_0.jpg')):
    do_unet = model.get_do_unet()

    # Check for existing weights
    if not os.path.exists(f'models/{model_name}_best.h5'):
        train('Test_scale')

    # load best weights
    do_unet.load_weights(f'models/{model_name}_best.h5')

    # load data
    img, mask, edge = data.load_data(image, padding=200)
    img_chips, mask_chips, edge_chips = data.test_chips(img, mask, edge=edge, padding=100)

    # segment all image chips
    output = do_unet.predict(img_chips)
    new_mask_chips = np.array(output[0])
    new_edge_chips = np.array(output[1])

    # reshape chips arrays to be concatenated
    new_mask_chips = reshape(new_mask_chips, get_sizes(img)[0][0], get_sizes(img)[0][1], 'output')
    new_edge_chips = reshape(new_edge_chips, get_sizes(img)[0][0], get_sizes(img)[0][1], 'output')

    # concatenate chips into a single image (mask and edge)
    new_mask = concat(new_mask_chips)
    new_edge = concat(new_edge_chips)

    # organize results into one figure
    fig = plt.figure(figsize=(25, 12), dpi=80)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Test image')
    ax.imshow(np.array(img)[0,:,:,:])
    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Test mask')
    ax.imshow(np.array(mask)[0,:,:])
    ax = fig.add_subplot(2, 3, 3)
    ax.set_title('Test edge')
    ax.imshow(np.array(edge)[0,:,:])
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Predicted mask')
    ax.imshow(new_mask)
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Predicted edge')
    ax.imshow(new_edge)

    # save the figure as a sample output
    plt.savefig('sample.png')


# evaluate model accuracies (mask accuracy and edge accuracy)
def evaluate():
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    do_unet = model.get_do_unet()

    # Check for existing weights
    if not os.path.exists(f'models/Test_scale_best.h5'):
        train('Test_scale')

    # load best weights
    do_unet.load_weights(f'models/Test_scale_best.h5')

    # load data
    img, mask, edge = data.load_data(test_img_files)
    img_chips, mask_chips, edge_chips = data.test_chips(img, mask, edge=edge)

    # print the evaluated accuracies
    print(do_unet.evaluate(img_chips, (mask_chips, edge_chips)))


# threshold image using otsu's threshold
def threshold(img = 'output/edge.png'):
    if not os.path.exists(f'{img}'):
        print('Image does not exist!')
        return

    img = cv2.imread(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

    # save the resulting thresholded image
    plt.imsave('output/threshold_edge.png', img)


if __name__ == '__main__':
    predict()
    threshold()
