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