# LeNet5 object
import numpy

class LeNet5(object):
    def __init__(self):

        self.w1 = numpy.random.randn(5, 5, 1, 6) / 25
        self.w3 = numpy.random.randn(5, 5, 6, 16) / 25
        self.w5 = numpy.random.randn(5, 5, 16, 120) / 25
        self.w6 = numpy.random.randn(120,84)
        self.w7 = numpy.random.randn(84, 10)

        self.b1 = numpy.zeros((1,1,1,self.w1.shape[-1]))
        self.b3 = numpy.zeros((1,1,1,self.w3.shape[-1]))
        self.b5 = numpy.zeros((1,1,1,self.w5.shape[-1]))
        self.b6 = numpy.zeros((1,self.w6.shape[-1]))
        self.b7 = numpy.zeros((1,self.w7.shape[-1]))

        #raise NotImplementedError

    def Forward_Propagation(self, input_image, input_label, mode):

        self.x = input_image
        self.z1,self.cachez1 = convolution(input_image, self.w1, 6, self.b1)

        self.z2,self.cachez2 = maxpooling(self.z1)
        self.a2 = relu(self.z2)

        self.z3,self.cachez3 = convolution(self.a2, self.w3, 16, self.b3)

        self.z4,self.cachez4 = maxpooling(self.z3)
        self.a4 = relu(self.z4)

        self.z5,self.cachez5 = convolution(self.a4, self.w5, 120, self.b5)
        self.a5 = relu(self.z5)

        self.a5_flatten = self.a5[:, 0, 0, :]
        self.z6 = fc(self.a5_flatten, self.w6, self.b6)
        self.a6 = relu(self.z6)

        self.z7 = fc(self.a6, self.w7, self.b7)
        self.a7 = self.softmax( self.z7)

        self.y = input_label

        if mode == "train":
            n_samples = input_label.shape[0]  # 256
            logp = - numpy.log(self.a7.T + 1e-8) * self.y
            loss = numpy.sum(logp) / n_samples
            return loss
        else:
            class_pred = numpy.argmax(self.a7, axis=1)
            error01 = numpy.sum(input_label != class_pred)
            return error01, class_pred
        #raise NotImplementedError

    def Back_Propagation(self, lr_global):

        weighted_y=onehot(self.y)
        a7_delta= self.crossentropy(self.a7, weighted_y)

        z6_delta=numpy.dot(a7_delta,self.w7.T)
        a6_delta=z6_delta * self.relu_deri(self.a6)

        z5_delta = numpy.dot(a6_delta, self.w6.T)

        z5_delta = z5_delta[:, numpy.newaxis, numpy.newaxis, :]
        a5_delta = z5_delta * self.relu_deri(self.a5)

        z4_delta = numpy.zeros(self.a4.shape)
        a5_dW = numpy.zeros(self.w5.shape)
        a5_db = numpy.zeros((1, 1, 1, 120))
        for h in range(1):
            for w in range(1):
                z4_delta[:, h:h+5, w:w+5, :] = numpy.transpose(numpy.dot(self.w5, a5_delta[:, h, w, :].T), (3, 0, 1, 2))
                a5_dW += numpy.dot(numpy.transpose(self.cachez5[:, h:h+5, w:w+5, :], (1, 2, 3, 0)), a5_delta[:, h, w, :])
                a5_db += numpy.sum(a5_delta, axis=0, keepdims=True)

        a4_delta = z4_delta * self.relu_deri(self.a4)

        a3_delta = maxpool_deri(a4_delta, self.cachez4)

        z2_delta = numpy.zeros(self.a2.shape)
        a3_dW = numpy.zeros(self.w3.shape)
        a3_db = numpy.zeros((1, 1, 1, 16))
        for h in range(10):
            for w in range(10):
                z2_delta[:, h:h+5, w:w+5, :] += numpy.transpose(numpy.dot(self.w3, a3_delta[:, h, w, :].T), (3, 0, 1, 2))
                a3_dW += numpy.dot(numpy.transpose(self.cachez3[:, h:h + 5, w:w + 5, :], (1, 2, 3, 0)),a3_delta[:, h, w, :])
                a3_db += numpy.sum(a3_delta[:,h,w,:], axis=0, keepdims=True)

        a2_delta = z2_delta * self.relu_deri(self.a2)

        a1_delta =  maxpool_deri(a2_delta,self.cachez2)

        z1_delta = numpy.zeros(self.x.shape)
        a1_dW = numpy.zeros(self.w1.shape)
        a1_db = numpy.zeros((1, 1, 1, 6))
        for h in range(28):
            for w in range(28):
                z1_delta[:, h:h + 5, w:w + 5, :] += numpy.transpose(numpy.dot(self.w1, a1_delta[:, h, w, :].T),(3, 0, 1, 2))
                a1_dW += numpy.dot(numpy.transpose(self.cachez1[:, h:h + 5, w:w + 5, :], (1, 2, 3, 0)), a1_delta[:, h, w, :])
                a1_db += numpy.sum(a1_delta[:, h, w, :], axis=0, keepdims=True)
        self.w7 -= lr_global * numpy.dot(self.a6.T,a7_delta)
        self.b7 -= lr_global * numpy.sum(a7_delta, axis=0, keepdims=True)

        self.a5_flatten = self.a5[:, 0, 0, :]
        self.w6 -= lr_global * numpy.dot(self.a5_flatten.T, a6_delta)
        self.b6 -= lr_global * numpy.sum(a6_delta, axis=0, keepdims=True)

        self.w5 -= lr_global * a5_dW
        self.b5 -= lr_global * a5_db

        self.w3 -= lr_global * a3_dW
        self.b3 -= lr_global * a3_db

        self.w1 -= lr_global * a1_dW
        self.b1 -= lr_global * a1_db

    def softmax(self, Z):
        expZ = numpy.exp(Z - numpy.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def crossentropy(self,pred,real):
        n_samples = real.shape[0]
        res = pred - real
        return res / n_samples

    def relu_deri(self,x):
        x[x <= 0] = 0
        return x

def onehot (y):
    shape=(y.size,y.max()+1)
    onehot=numpy.zeros(shape)
    rows=numpy.arange(y.size)
    onehot[rows,y]=1
    return onehot

def maxpooling(feature_map,size=2,stride=2):

    batch, _, in_dim, depth = feature_map.shape
    out_dim = int((in_dim - size) / stride) + 1

    pool = numpy.zeros((batch, out_dim, out_dim, depth))

    for h in range(out_dim):
        for w in range(out_dim):
            pool[:, h, w, :] = numpy.max(feature_map[:, h * stride:h * stride + size, w * stride:w * stride + size, :], axis=(1, 2))
    cache=feature_map
    return pool,cache

def maxpool_deri(feature_map,cache):

    x = cache
    batch,HH,WW,depth=feature_map.shape

    dx=None
    dx=numpy.zeros(x.shape)
    for n in range(batch):
        for depth in range(depth):
            for r in range(HH):
                for c in range(WW):
                    x_pool=x[n,r*2:r*2+2,c*2:c*2+2,depth]
                    mask=(x_pool==numpy.max(x_pool))
                    dx[n,r*2:r*2+2,c*2:c*2+2,depth]=mask*feature_map[n,r,c,depth]
    return dx

def relu(feature_map):
    reluout = numpy.where(feature_map>0, feature_map, 0)
    return reluout

def fc(feature_map, weight, bias):
    return numpy.dot(feature_map, weight) + bias

def convolution(input_image, filt, no_filter, bias, filter_size=5, stride=1):

    batch, input_dim, _, depth = input_image.shape

    out_dim = int((input_dim - filter_size) / stride) + 1
    convout = numpy.zeros((batch,out_dim, out_dim, no_filter))

    height = 0
    while height + filter_size <= input_dim:
        width = 0
        while width + filter_size <= input_dim:
            convout[:, height ,width, :] = numpy.tensordot((input_image[:, height:height + filter_size, width:width + filter_size,:]), filt, axes=([1,2,3],[0,1,2])) + bias
            width += stride
        height += stride
    cache = input_image
    return convout, cache
