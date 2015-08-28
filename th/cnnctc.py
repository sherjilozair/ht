import cPickle
import numpy, theano
from theano import tensor as T
datapath = 'data.small.pkl'
with open(datapath) as f:
    data = cPickle.load(f)

images = data['x']
labels = data['y']
chars = data['chars']
chars.append("<blank>")
n_classes = len(chars)

X = T.tensor4('image')
Y = T.ivector('label')
lr = T.scalar('lr')

# 1, 1, 64, 1208
W1 = theano.shared(numpy.random.randn(100, 1, 64, 32) * 0.001)
B1 = theano.shared(numpy.zeros((100,)))
H1 = T.nnet.conv2d(X, W1, subsample=(64, 1)) + B1.dimshuffle('x', 0, 'x', 'x')
# 1, 100, 1, 1208 - 32 + 1
W2 = theano.shared(numpy.random.randn(100, 100, 1, 1) * 0.001)
B2 = theano.shared(numpy.zeros((100,)))
H2 = T.nnet.conv2d(H1, W2, subsample=(1, 1)) + B2.dimshuffle('x', 0, 'x', 'x')

W3 = theano.shared(numpy.random.randn(n_classes, 100, 1, 1) * 0.001)
B3 = theano.shared(numpy.zeros((n_classes,)))
H4 = T.nnet.conv2d(H2, W3, subsample=(1, 1)) + B3.dimshuffle('x', 0, 'x', 'x')

H5 = H4[0, :, 0, :]

YP = T.nnet.softmax(H5.T)

from ctccost import ctc

cost = ctc(YP, Y)

params = [W1, B1, W2, B2, W3, B3]
grads = T.grad(cost, params)
updates = [(p, p - lr * g) for p, g in zip(params, grads)]

train = theano.function([X, Y, lr], [cost, YP], updates=updates, allow_input_downcast=True)

def label_seq(string):
    idxs = map(lambda w: chars.index(w), string)
    result = numpy.ones((len(idxs)*2 + 1,),dtype=numpy.int32) * (len(chars) - 1)
    result[numpy.arange(len(idxs))*2+1] = idxs

    return result


for n in xrange(1000):
	total_cost = 0.
	for i in xrange(len(images)):
            img = images[i].reshape(1, 1, images[i].shape[0], images[i].shape[1])
            label = labels[i].replace(' ', '|')
            label = label_seq(label)
            img = img.astype('float32')
            label = label.astype('int32')
            [cost, y] = train(img, label, 0.001)
            total_cost += cost
            if i < 3:
                print "shown:", labels[i]
		print "seen:", "".join(map(lambda i: chars[i], y.argmax(axis=1)))
		print cost
	print "epoch", n, total_cost / len(images)
