'''
Author: Ruida Wang
'''
import numpy as np
import random

# This is the class for DataLoader in the model, the data should be in form
class Data:
    def __init__(self,
                 dataset,
                 tag,
                 train=True,
                 batch_size=256,
                 shuffle=False,
                 original_data=None):

        self.tag = tag
        self.train=train
        self.dataNum = len(dataset[0])
        self.input = dataset[0]
        self.output = dataset[1]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.original_data = original_data

        self.idx_ref = list(range(self.dataNum))

        if shuffle:
            random.shuffle(self.idx_ref)
            self.input = [self.input[i] for i in self.idx_ref]
            self.output = [self.output[i] for i in self.idx_ref]

            if self.original_data != None:
                self.original_data[0] = [self.original_data[0][i] for i in self.idx_ref]
                self.original_data[1] = [self.original_data[1][i] for i in self.idx_ref]

    def shuffle(self):
        random.shuffle(self.idx_ref)
        self.input = [self.input[i] for i in self.idx_ref]
        self.output = [self.output[i] for i in self.idx_ref]



    # it will generated batched result of the data, the batch will be organized as a list
    # the format will be [batch_1, batch_2, ...]
    # batch_i = [data_i, label_i]
    # data_i will be a (batch_size, max_len, hidden_size) shape matrix, label_i will be a (batch_size, ) shape matrix
    def generate_batch(self):
        to_return = []
        batch_num = int(self.dataNum/self.batch_size)
        for i in range(batch_num):
            to_return += [[self.input[i * self.batch_size : (i + 1) * self.batch_size],
                           self.output[i * self.batch_size : (i + 1) * self.batch_size]]]

        to_return += [[self.input[batch_num * self.batch_size : ],
                       self.output[batch_num * self.batch_size : ]]]

        for i in range(len(to_return)):
            for j in range(2):
                to_return[i][j] = np.array(to_return[i][j])

        return to_return

    def get_original(self, idxLs):
        to_return = [[], []]
        for i in idxLs:
            to_return[0].append(self.original_data[0][i])
            to_return[1].append(self.original_data[1][i])
        return to_return