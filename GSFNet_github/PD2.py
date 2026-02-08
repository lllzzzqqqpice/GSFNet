
import numpy as np

num_classes = 6

PD_CLASSES = ['Invalid', 'Impervious surfaces', 'Building', 'Low vegetation',
              'Tree', 'Car', 'Clutter/background']
PD_MEAN = np.array([85.8, 91.7, 84.9, 96.6, 47])
PD_STD = np.array([35.8, 35.2, 36.5, 37, 55])

PD_COLORMAP = [ [255, 255, 255],[255, 255, 0],[0, 255, 0], [0, 0, 255], [0, 255, 255],
                 [255, 0, 0],[0, 0, 0]]
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(PD_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Index2Color(pred):
    colormap = np.asarray(PD_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]
