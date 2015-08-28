from scipy import misc
import os
from PIL import Image
import scipy.ndimage
import numpy
import cPickle

mean_com = 57.86 # all images should be centered around this value.

def retrieve_image(name):
    [a, b, c] = name.split('-')
    path = "lines/%(a)s/%(a)s-%(b)s/%(a)s-%(b)s-%(c)s.png" % locals()
    return misc.imread(path)

def binarize(img, binarizer):
    return img > binarizer

def invert(img):
    return 255 - img

def resize(img, xsz):
    ysz = int(round(img.shape[1] * float(xsz) / img.shape[0]))
    return misc.imresize(img, (xsz, ysz))

def compute_std(img, mean):
    y = (numpy.arange(img.shape[0]).reshape(-1, 1) - mean) ** 2
    return numpy.sqrt((y * img).sum() / img.sum())

chars = set()
if __name__ == '__main__':
    lines = filter(lambda s: not s.startswith('#'), open('ascii/lines.txt').read().split('\n'))[:-1]
    images = []
    labels = []
    shapes = []
    coms = []
    for i, line in enumerate(lines):
        if i == 100:
            break
        [name, status, binarizer, num_components, x, y, w, h, label] = line.split(' ', 8)
        img = retrieve_image(name)
        img = 255 - img
        com = scipy.ndimage.measurements.center_of_mass(img)[0]
        std = compute_std(img, com)
        img = img[max(0, int(com - 3 * std)): min(img.shape[1], int(com + 3 * std))]
        coms.append(com)
        img = resize(img, 64)
        img = binarize(img, float(binarizer))
        for c in label:
            chars.add(c)
        shapes.append(img.shape)
        images.append(img)
        labels.append(label)
    with open('data.small.pkl', 'w') as f:
        cPickle.dump({'x':images, 'y': labels, 'chars': list(chars)}, f)

