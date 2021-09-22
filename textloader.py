# The MIT License (MIT)
# 
# Copyright (c) 2015 Sherjil Ozair, 2018 UMONS
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
import os
import sys
import codecs
from six.moves import cPickle
import numpy as np
import collections


class TextLoader:
    def __init__(self, batch_size=0, seq_length=0, preprocess_dir="save/preprocess", config_name="config",
                 index_name="index", tensor_name="data", encoding=None):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.config_name = config_name
        self.preprocess_dir = preprocess_dir
        self.index_name = index_name
        self.tensor_name = tensor_name
        self.encoding = encoding

        if not os.path.exists(self.preprocess_dir):
            print("No preprocessed files.")
            sys.exit(-1)

        self.vocab = None
        self.vocab_size = 0
        self.words = None
        self.tensor = None

        self.num_batches = 0
        self.x_batches = None
        self.y_batches = None
        self.pointer = 0

        # print("Loading preprocessed files from " + self.preprocess_dir + "...")
        # self.load_preprocessed()
        #
        # self.create_batches()
        # self.reset_batch_pointer()

    def load_data(self, corpus_list=None):
        data = ""

        if corpus_list is None:
            corpus_list = []
            for name in os.listdir(self.preprocess_dir):
                if os.path.isdir(os.path.join(self.preprocess_dir, name)):
                    corpus_list.append(name)

        for name in corpus_list:
            preprocess_corpus = os.path.join(self.preprocess_dir, name)

            if not os.path.isdir(preprocess_corpus):
                continue

            print("Subdir: " + name)

            for file in os.listdir(preprocess_corpus):
                path = os.path.join(preprocess_corpus, file)

                if not os.path.isfile(path) \
                        or os.path.basename(path) == self.config_name + ".json"\
                        or os.path.basename(path) == self.index_name + ".txt":
                    continue

                print("File: " + file)
                with codecs.open(path, "r", encoding=self.encoding) as f:
                    temp = f.read()
                data += temp + " \n "

        return data.strip().split(" ")

    def load_index(self, data):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter()
        word_counts.update(data)

        # Mapping from index to word
        index = [x[0] for x in word_counts.most_common()]
        index = list(sorted(index))

        return index

    def load_preprocessed(self):
        words_file = os.path.join(self.preprocess_dir, self.index_name + ".pkl")
        tensor_file = os.path.join(self.preprocess_dir, self.tensor_name + ".npy")

        with open(words_file, 'rb') as f:
            self.words = cPickle.load(f)

        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
