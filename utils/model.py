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
        # self.w5 = numpy.random.randn(5, 5, 16, 120) / 25
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
        self.x=input_image
        self.z1 = convolution(input_image, self.w1, 6, self.b1)

        # print("z1", self.z1.shape)
        # Layer S2 followed by ReLu activation
        self.z2,self.cachez2 = maxpooling(self.z1)
        self.a2 = relu(self.z2)
        # print("a2", self.a2.shape)
        # print("a2", self.a5.shape)
        # Layer C3
        self.z3 = convolution(self.a2, self.w3, 16, self.b3)
        # print("z3", self.z3.shape)
        # Layer S4 followed by ReLu activation
        self.z4,self.cachez4 = maxpooling(self.z3)
        self.a4 = relu(self.z4)
        # print("a4", self.a4.shape)
        # Layer C5 followed by ReLu activation
        self.z5 = convolution(self.a4, self.w5, 120, self.b5)
        self.a5 = relu(self.z5)
        self.a5_flatten = self.a5[:, 0, 0, :]
        # Layer F6  followed by ReLu activation
        self.z6 = fc(self.a5_flatten, self.w6, self.b6)
        # self.a6.resize(84,1,1)
        self.a6 = relu(self.z6)
        # print("a6",self.a6.shape)
        # Layer F7
        self.z7 = fc(self.a6, self.w7, self.b7)
        # print("a7",self.z7.shape)

        print("Before softmax: ", self.z7.shape)
        self.a7 = self.softmax( self.z7)
        print("After softmax", self.a7.shape)

        self.y = input_label
        # self.y.resize((256, 10))
        # print(self.y)
        n_samples = input_label.shape[0] #256
        #print("After softmax: ", self.out[:3])
        self.class_pred = numpy.argmax( self.a7, axis=1)
        #print(self.class_pred)
        #print("predicted label",self.class_pred)

        if mode == "train":
            logp = - numpy.log(numpy.argmax(self.a7, axis=1))
            loss = numpy.sum(logp) / n_samples
            print("loss: ", loss)
            return loss
        else:
            error01 = numpy.sum(input_label != self.class_pred)
            print("error01: ", error01)
            return error01, self.class_pred

        #raise NotImplementedError

    def Back_Propagation(self, lr_global):
        # YOUR IMPLEMETATION
        print("input_labels: ",self.y.shape)

        weighted_y = self.w7[self.y,:]
        print("weighted_label: " ,weighted_y.shape)

        a7_delta= self.crossentropy(self.a7, weighted_y)
        print("a7_delta.shape after crossentropy: ", a7_delta.shape)

        z6_delta=numpy.dot(a7_delta,self.w7.T)
        print("z6_delta.shape: ", z6_delta.shape)

        a6_delta=z6_delta * self.relu_deri(self.a6)
        print("a6_delta.shape: ",a6_delta.shape)

        z5_delta = numpy.dot(a6_delta, self.w6.T)
        print("z5_delta.shape: ", z5_delta.shape)

        z5_delta = z5_delta[:, numpy.newaxis, numpy.newaxis, :]
        print("After reverse flatten z5_delta.shape: ", z5_delta.shape)
        a5_delta = z5_delta * self.relu_deri(self.a5)
        print("a5_delta.shape: ", a5_delta.shape)

        z4_delta = numpy.zeros((25,5,5,16))

        print("w5.shape: ", self.w5.shape)
        for h in range(1):
            for w in range(1):
                z4_delta[:, h:h+5, w:w+5, :] += numpy.transpose(numpy.dot(self.w5, a5_delta[:, h, w, :].T), (3,0,1,2))
        print("z4_delta.shape: ", z4_delta.shape)

        print("After transpose z4_delta.shape: ", z4_delta.shape)

        a4_delta = z4_delta * self.relu_deri(self.a4)
        print("a4_delta.shape: ", a4_delta.shape)

        a3_delta = maxpool_deri(a4_delta, self.cachez4)
        print("a3_delta.shape: ",a3_delta.shape)

        z4_delta = z5_delta
        # YOUR IMPLEMETATION
        print("input_labels: ",self.y.shape)

        weighted_y = self.w7[self.y,:]
        print("weighted_label: " ,weighted_y.shape)

        a7_delta= self.crossentropy(self.a7, weighted_y)
        print("a7_delta.shape after crossentropy: ", a7_delta.shape)

        z6_delta=numpy.dot(a7_delta,self.w7.T)
        print("z6_delta.shape: ", z6_delta.shape)

        a6_delta=z6_delta * self.relu_deri(self.a6)
        print("a6_delta.shape: ",a6_delta.shape)

        z5_delta = numpy.dot(a6_delta, self.w6.T)
        print("z5_delta.shape: ", z5_delta.shape)

        z5_delta = z5_delta[:, numpy.newaxis, numpy.newaxis, :]
        print("After reverse flatten z5_delta.shape: ", z5_delta.shape)
        a5_delta = z5_delta * self.relu_deri(self.a5)
        print("a5_delta.shape: ", a5_delta.shape)

        z4_delta = numpy.zeros((25, 5, 5, 16))
        print("w5.shape: ", self.w5.shape)
        for h in range(1):
            for w in range(1):
                z4_delta[:, h:h+5, w:w+5, :] = numpy.transpose(numpy.dot(self.w5, a5_delta[:, h, w, :].T), (3, 0, 1, 2))
        print("z4_delta.shape: ", z4_delta.shape)

        a4_delta = z4_delta * self.relu_deri(self.a4)
        print("a4_delta.shape: ", a4_delta.shape)

        a3_delta = maxpool_deri(a4_delta, self.cachez4)
        print("a3_delta.shape: ",a3_delta.shape)

        z2_delta = numpy.zeros((25, 14, 14, 6))

        print("w3.shape: ", self.w3.shape)
        for h in range(10):
            for w in range(10):
                z2_delta[:, h:h+5, w:w+5, :] += numpy.transpose(numpy.dot(self.w3, a3_delta[:, h, w, :].T), (3, 0, 1, 2))
        print("z2_delta.shape: ", z2_delta.shape)

        print("a2.shape: ", self.a2.shape)
        a2_delta = z2_delta * self.relu_deri(self.a2)
        print("a2_delta.shape: ", a2_delta.shape)
        a1_delta =  maxpool_deri(a2_delta,self.cachez2)
        print("a1_delta.shape: ", a1_delta.shape)

        print("weight update w7: ", self.a6.T.shape, a7_delta.shape)
        self.w7 -= lr_global * numpy.dot(self.a6.T,a7_delta)
        self.b7 -= lr_global * numpy.sum(a7_delta, axis=0, keepdims=True)

        print("weight update w6: ", self.a5.T.shape, a6_delta.shape)
        self.w6 -= lr_global * numpy.dot(self.a5.T, a6_delta)
        self.b6 -= lr_global * numpy.sum(a6_delta, axis=0, keepdims=True)

        self.w5 -= lr_global * numpy.dot(self.a4.T, a5_delta)
        self.b5 -= lr_global * numpy.sum(a5_delta, axis=0, keepdims=True)

        self.b3 -= lr_global * numpy.sum(a3_delta, axis=0, keepdims=True)
        self.w3 -= lr_global * numpy.dot(self.a2.T, a3_delta)

        self.b1 -= lr_global * numpy.sum(a1_delta, axis=0, keepdims=True)
        self.w1 -= lr_global * numpy.dot(self.x.T, a1_delta)


    def softmax(self, Z):
        expZ = numpy.exp(Z - numpy.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def crossentropy(self,pred,real):
        # res = pred - real
    #         # return res
        n_samples = real.shape[0]
        res = pred - real
        return res / n_samples

    def relu_deri(self,Z):
        return numpy.where(Z <= 0, 0, 1)
def maxpooling(feature_map,size=2,stride=2):
    #declare an empty array for storing the output
    # pool=numpy.zeros(((feature_map.shape[0]-size+1)/stride),
    #                  ((feature_map.shape[1]-size+1)/stride),
    #                  (feature_map.shape[-1]))
    #pool = numpy.zeros(((14,14,6)))

    batch, _, in_dim, depth = feature_map.shape
    # print("feature",feature_map.shape)
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
    cache=feature_map
    # print("pool.shape:", pool.shape)
    return pool,cache

def maxpool_deri(feature_map,cache):
    x = cache
    batch, H, W, depth = x.shape
    batch,HH,WW,depth=feature_map.shape
    # print("x.shape",x.shape)
    # print("feature.shape", feature_map.shape)

    dx=None
    dx=numpy.zeros(x.shape)
    for n in range(batch):
        for depth in range(depth):
            for r in range(HH):
                for c in range(WW):
                    x_pool=x[n,r*2:r*2+2,c*2:c*2+2,depth]
                    # print("x_pool",x_pool)
                    mask=(x_pool==numpy.max(x_pool))
                    dx[n,r*2:r*2+2,c*2:c*2+2,depth]=mask*feature_map[n,r,c,depth]

    return dx

def relu(feature_map):
    #reluout=numpy.zeros(feature_map.shape)
    #for map_num in range(feature_map.shape[-1]):
    #    for r in numpy.arange(0,feature_map.shape[1]):
    #        for c in numpy.arange(0,feature_map.shape[2]):
    #            reluout[:,r,c,map_num]=numpy.max([feature_map[:,r,c,map_num]],0)
    reluout = numpy.where(feature_map>0, feature_map, 0)
    # print("reluout.shape:", reluout.shape)
    return reluout

def fc(feature_map, weight, bias):
    # print("fc layer feature_map.shape:", feature_map.shape)
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
