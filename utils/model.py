# LeNet5 object
import numpy

class LeNet5(object):
    def __init__(self):
        # YOUR IMPLEMETATION
        #define our layers
        #I guess there is no weight for maxpooling layer!
        self.w1 = numpy.random.randn(5, 5, 1, 6) / 25
        #self.w2 = numpy.random.randn(14, 14) / 196
        self.w3 = numpy.random.randn(5, 5, 6, 16) / 25
        #self.w4 = numpy.random.randn(5, 5) / 25
        self.w5 = numpy.random.randn(5, 5, 16, 120) / 25
        self.w6 = numpy.random.randn(120,84)
        self.w7 = numpy.random.randn(84, 10)

        self.b1 = numpy.zeros((1,1,1,self.w1.shape[-1]))
        #self.b2 = numpy.zeros((1,1,1,self.w2.shape[-1]))
        self.b3 = numpy.zeros((1,1,1,self.w3.shape[-1]))
        #self.b4 = numpy.zeros((1,1,1,self.w4.shape[-1]))
        self.b5 = numpy.zeros((1,1,1,self.w5.shape[-1]))
        self.b6 = numpy.zeros((1,self.w6.shape[-1]))
        self.b7 = numpy.zeros((1,self.w7.shape[-1]))

        print("Init done")
        #raise NotImplementedError

    def Forward_Propagation(self, input_image, input_label, mode):
        # YOUR IMPLEMETATION
        # Layer C1
        self.a1 = convolution(input_image, self.w1, 6, self.b1)
        # Layer S2 followed by ReLu activation
        self.a2 = maxpooling(self.a1)
        self.a2 = relu(self.a2)
        #self.a2 = numpy.dot(self.a2, self.w2) + self.b2
        # Layer C3
        self.a3 = convolution(self.a2, self.w3, 16, self.b3)
        # Layer S4 followed by ReLu activation
        self.a4 = maxpooling(self.a3)
        self.a4 = relu(self.a4)
        #self.a4 = numpy.dot(self.a4, self.w4) + self.b4
        # Layer C5 followed by ReLu activation
        self.a5 = convolution(self.a4, self.w5, 120, self.b5)
        self.a5 = relu(self.a5)

        self.a5 = self.a5[:, 0, 0, :]

        # Layer F6  followed by ReLu activation
        self.a6 = fc(self.a5, self.w6, self.b6)
        self.a6 = relu(self.a6)
        # Layer F7
        self.a7 = fc(self.a6, self.w7, self.b7)

        #print("Before softmax: ", self.a7[:3])
        self.out = self.softmax(self.a7)

        self.y = input_label
        n_samples = input_label.shape[0] #256
        #print("After softmax: ", self.out[:3])
        class_pred = numpy.argmax(self.out, axis=1)

        if mode == "train":
            logp = - numpy.log(numpy.argmax(self.a7, axis=1))
            loss = numpy.sum(logp) / n_samples
            print("loss: ", loss)
            return loss
        else:
            error01 = numpy.sum(input_label != class_pred)
            print("error01: ", error01)
            return error01, class_pred

        #raise NotImplementedError

    def Back_Propagation(self, lr_global):
        # YOUR IMPLEMETATION
        a7_delta= self.crossentropy(self.a7,self.y)
        z6_delta=numpy.dot(a7_delta,self.w7)
        a6_delta=z6_delta*self.relu_deri(self,self.a6)
        z5_delta = numpy.dot(a6_delta, self.w6)
        a5_delta = z5_delta * self.relu_deri(self, self.a5)
        z4_delta = numpy.dot(a5_delta, self.w5)
        a4_delta = z4_delta * self.relu_deri(self, self.a4)
        z3_delta = a4_delta
        a3_delta = z3_delta * self.relu_deri(self, self.a3)
        z2_delta = numpy.dot(a3_delta, self.w3)
        a2_delta = z2_delta * self.relu_deri(self, self.a2)
        z1_delta = a2_delta
        a1_delta = z1_delta * self.relu_deri(self, self.x)

        self.w7 -= lr_global* numpy.dot(self.a6, a7_delta)
        self.w6 -= lr_global * numpy.dot(self.a5, a6_delta)
        self.w5 -= lr_global* numpy.dot(self.a4, a5_delta)
        self.w4 -= lr_global * numpy.dot(self.a3, a4_delta)
        self.w3 -= lr_global * numpy.dot(self.a2, a3_delta)
        self.w2 -= lr_global * numpy.dot(self.a1, a2_delta)
        self.w1 -= lr_global* numpy.dot(self.x, a1_delta)

    def softmax(self, Z):
        expZ = numpy.exp(Z - numpy.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def crossentropy(pred,real):
        res = pred - real
        return res

    def relu_deri(self,Z):
        if (Z>0):
            return 1
        else:
            return 0

def maxpooling(feature_map,size=2,stride=2):
    #declare an empty array for storing the output
    # pool=numpy.zeros(((feature_map.shape[0]-size+1)/stride),
    #                  ((feature_map.shape[1]-size+1)/stride),
    #                  (feature_map.shape[-1]))
    #pool = numpy.zeros(((14,14,6)))

    batch, _, in_dim, depth = feature_map.shape
    out_dim = int((in_dim - size) / stride) + 1

    pool = numpy.zeros((batch, out_dim, out_dim, depth))

    for map_num in range(depth):
        r2=0
        for r in numpy.arange(0,out_dim-size-1,stride):
            c2=0
            for c in numpy.arange(0,out_dim-size-1,stride):
                pool[:,r2,c2,map_num]=numpy.max([feature_map[:,r:r+size,c:c+size,map_num]])
                c2=c2+1
        r2=r2+1
    print("pool.shape:", pool.shape)
    return pool

def relu(feature_map):
    #reluout=numpy.zeros(feature_map.shape)
    #for map_num in range(feature_map.shape[-1]):
    #    for r in numpy.arange(0,feature_map.shape[1]):
    #        for c in numpy.arange(0,feature_map.shape[2]):
    #            reluout[:,r,c,map_num]=numpy.max([feature_map[:,r,c,map_num]],0)
    reluout = numpy.where(feature_map>0, feature_map, 0)
    print("reluout.shape:", reluout.shape)
    return reluout

def fc(feature_map, weight, bias):
    print("fc layer feature_map.shape:", feature_map.shape)
    return numpy.dot(feature_map, weight) + bias

# convolution layer
def convolution(input_image, filt, no_filter, bias, filter_size=5, stride=1):
    #print("conv input_image.shape: ", input_image)
    batch, input_dim, _, depth = input_image.shape  # image dimensions

    out_dim = int((input_dim - filter_size) / stride) + 1  # calculate output dimensions
    convout = numpy.zeros((batch,out_dim, out_dim, no_filter))

    # convolve each filter over the image
    #for f in range(no_filter):
    height = 0
    # move filter vertically across the image
    while height + filter_size <= input_dim:
        width = 0
        # move filter horizontally across the image
        while width + filter_size <= input_dim:
            # perform the convolution operation and add the bias
            convout[:, height ,width, :] = numpy.tensordot((input_image[:, height:height + filter_size, width:width + filter_size,:]), filt, axes=([1,2,3],[0,1,2])) + bias
            width += stride
        height += stride
    #print("convout.shape: ", convout.shape)
    return convout
