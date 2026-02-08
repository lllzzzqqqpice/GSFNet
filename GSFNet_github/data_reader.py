from osgeo import gdal
import random
import numpy as np


def read_image(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError('Image read error!')
    image = dataset.ReadAsArray().astype('float32') / 127.5 - 1.0
    image = np.transpose(image, [1, 2, 0])
    return image


def read_label(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError('Image read error!')
    label = dataset.ReadAsArray()
    label[label == 2] = 0
    label[label == 5] = 1
    label[label == 6] = 2
    label[label == 9] = 3
    label[label == 17] = 4
    label[label == 65] = 5
    return label


def read_disp(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise ValueError('Image read error!')
    disp = dataset.ReadAsArray()
    disp = np.expand_dims(disp, -1)
    return disp

# from read.data_argu import  random_crop,rand_flip
# def read_batch(left_paths, right_paths, label_paths, disp_paths):
#     lefts, rights, labels, disps = [], [], [], []
#     for left_path, right_path, label_path, disp_path in zip(left_paths, right_paths, label_paths, disp_paths):
#
#         left_path=read_image(left_path)
#         right_path=read_image(right_path)
#         label_path = read_label(label_path)
#         disp_path = read_disp(disp_path)
#         left_path, right_path, label_path, disp_path= random_crop(left_path, right_path, label_path, disp_path , [448,448])
#
#         left_path, right_path, label_path, disp_path = rand_flip(left_path, right_path, label_path, disp_path)
#
#         lefts.append(left_path)
#         rights.append(right_path)
#         labels.append(label_path)
#         disps.append(disp_path)
#
#     return np.array(lefts), np.array(rights), np.array(labels), np.array(disps)

def read_batch(left_paths, right_paths, label_paths, disp_paths):
    lefts, rights, labels, disps = [], [], [], []
    for left_path, right_path, label_path, disp_path in zip(left_paths, right_paths, label_paths, disp_paths):
        lefts.append(read_image(left_path))
        rights.append(read_image(right_path))
        labels.append(read_label(label_path))
        disps.append(read_disp(disp_path))

    return np.array(lefts), np.array(rights), np.array(labels), np.array(disps)



def load_batch(all_left_paths, all_right_paths, all_label_paths, all_disp_paths, batch_size, reshuffle):
    # print(len(all_left_paths))
    # print(len(all_right_paths))
    # print(len(all_disp_paths))
    # print(len(all_label_paths))
    # print(all_left_paths[0])
    # print(all_right_paths[0])
    # print(all_label_paths[0])
    # print(all_disp_paths[0])
    assert len(all_left_paths) == len(all_right_paths)
    assert len(all_left_paths) == len(all_label_paths)
    assert len(all_left_paths) == len(all_disp_paths)

    print('load_batch')
    i = 0
    while True:

        lefts, rights, labels, disps = read_batch(
            all_left_paths[i*batch_size:(i+1)*batch_size],
            all_right_paths[i*batch_size:(i+1)*batch_size],
            all_label_paths[i*batch_size:(i+1)*batch_size],
            all_disp_paths[i*batch_size:(i+1)*batch_size])
        # # print(type(lefts))
        # print(lefts.shape)
        # # print(type(rights))
        # print(rights.shape)
        # # print(type(labels))
        # print(labels.shape)
        # # print(type(disps))
        # print(disps.shape)

        print('/n image')

        yield [lefts, rights],[labels,labels,disps,disps]
        # yield [lefts, rights],
        # yield  [labels ,labels, disps, disps]
        print('/n jiancha')
        i = (i + 1) % (len(all_left_paths) // batch_size)
        print('load_batch med')
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_label_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_label_paths, all_disp_paths = zip(*paths)
                print('load_batch done')
