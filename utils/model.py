# LeNet5 object
import numpy


class LeNet5(object):

    def maxpooling (feature_map,size=2,stride=2):
        #declare an empty array for storing the output
        pool=numpy.zeros(numpy.unit16((feature_map.shape[0]-size+1)/stride),
                         numpy.uint16((feature_map.shape[1]-size+1)/stride),
                         numpy.unit16(feature_map.shape[-1]))
        for map_num in feature_map.shape[-1]:
            r2=0
            for r in numpy.arrange(0,feature_map.shape[0]-size-1,stride):
                c2=0
                for c in numpy.arrange(0,feature_map.shape[1]-size-1,stride):
                    pool[r2,c2,map_num]=numpy.max([feature_map[r:r+size,c:c+size,map_num]])
                    c2=c2+1
            r2=r2+1
        return pool
    def relu(feature_map):
        reluout=numpy.zero(feature_map.shape)
        for map_num in feature_map.shape[-1]:
            for r in numpy.arrange(0,feature_map.shape[0]):
                for c in numpy.arrange(0,feature_map.shape[1]):
                    reluout[r,c,map_num]=numpy.max([feature_map[r,c,map_num]])

        return reluout


    def fc6(feature_map):
        return feature_map.reshape(84,1)

    def fc7(feature_map):
        return feature_map.reshape(10,1)


    # convolution layer
    def convolution(input_image, no_filter, filter_size=5, s=1):

        filt = numpy.random.randn(no_filter, filter_size, filter_size) / 25
        depth, input_dim, _ = input_image.shape  # image dimensions
        out_dim = int((input_dim - filter_size) / s) + 1  # calculate output dimensions
        #print(out_dim)
        convout = numpy.zeros((no_filter, out_dim, out_dim))

        # convolve each filter over the image
        for f in range(no_filter):
            height = 0
            # move filter vertically across the image
            while height + filter_size <= input_dim:
                width = 0
                # move filter horizontally across the image
                while width + filter_size <= input_dim:
                    # perform the convolution operation and add the bias
                    convout[f, height, width] = numpy.sum(filt[f] * input_image[:, height:height+filter_size, width:width+filter_size])
                    width += s
                height += s

        return convout








    def __init__(self):
        # YOUR IMPLEMETATION
        #define our layers
        raise NotImplementedError

    def Forward_Propagation(self, input_image, input_label, mode):
        # YOUR IMPLEMETATION
        raise NotImplementedError

    def Back_Propagation(self, lr_global):
        # YOUR IMPLEMETATION
        raise NotImplementedError