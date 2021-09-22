# The MIT License (MIT)
# 
# Copyright (c) 2018 UMONS
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

from __future__ import print_function
import tensorflow as tf

import os
from six.moves import cPickle
import codecs

from model import Model


class Generators:
    def __init__(self, save_dir, encoding, models=[], max_load=1):
        self.generators = {}
        self.loaded = []  # Last loaded is last of the list
        self.max_load = max_load

        if len(models) == 0:
            for model in os.listdir(save_dir):
                if os.path.isdir(os.path.join(save_dir, model)):
                    models.append(model)

        for name in models:
            self.generators[name] = Generator(save_dir, encoding, name)

        for i in range(max_load):
            self.load(models[i])

    def load(self, name):
        self.generators[name].load()
        self.loaded += [name]

    def unload(self, name):
        self.generators[name].close()
        self.loaded.remove(name)
        # Clean if there is no more load
        if len(self.loaded) == 0:
            tf.reset_default_graph()

    def unload_old(self):
        if len(self.loaded) > 0:
            self.unload(self.loaded[0])

    # To be called when you used a generator (update loaded list)
    def refresh(self, name=""):
        if name in self.loaded:
            self.loaded.remove(name)
            self.loaded += [name]

    def generate(self, name, n_tokens, prime, sample, pick, width, quiet, suppress_prime, towards_style=None):
        if name not in self.loaded:
            if len(self.loaded) == self.max_load:
                self.unload_old()
            self.load(name)

        self.refresh(name)

        if towards_style is not None:
            unk_vocab = [x for x in self.generators[name].words if x not in self.generators[towards_style].words]
        else:
            unk_vocab = None
        return self.generators[name].generate(n_tokens, prime, sample, pick, width, quiet, suppress_prime, unk_vocab)

    def get_percent_restricted_vocab(self, style1, style2):
        unk_vocab = [x for x in self.generators[style1].get_words() if x not in self.generators[style2].get_words()]
        return (1 - len(unk_vocab)/len(self.generators[style1].get_words())) * 100

    def change_words(self, name, string):
        return self.generators[name].change_words(string)

    def get_names(self):
        return [self.generators[x].long_name for x in self.generators]

    def get_loaded(self):
        return self.generators[self.loaded].long_name

    def close(self):
        for name in self.loaded:
            self.generators[name].close()
            self.loaded = []


class Generator:
    def __init__(self, save_dir, encoding, hash_name, long_name=None, config_name="config", index_name="index",
                 preprocess_name="preprocess", match_table_name="match_table"):
        self.is_loaded = False
        self.hash_name = hash_name
        self.long_name = long_name if long_name is not None else hash_name
        self.dir = os.path.join(save_dir, self.hash_name)
        with open(os.path.join(self.dir, config_name + '.pkl'), 'rb') as f:
            self.args = cPickle.load(f)
        with open(os.path.join(self.dir, os.path.join(preprocess_name, index_name + '.pkl')), 'rb') as f:
            self.words = cPickle.load(f)
            self.vocab = dict(zip(self.words, range(len(self.words))))

        self.low_val = next(i for i, word in enumerate(self.words) if word.startswith("_APPEND_"))
        self.high_val = len(self.words) - next(
            i for i, word in enumerate(reversed(self.words)) if word.startswith("_APPEND_"))

        self.match_words = dict()
        table_path = os.path.join(self.dir, os.path.join(preprocess_name, match_table_name + '.txt'))
        if os.path.exists(table_path):
            with codecs.open(table_path, 'r', encoding=encoding) as f:
                for line in f:
                    words = line.split(" ")
                    if len(words) < 2:
                        continue
                    key = words[0].strip()
                    value = " ".join(words[1:]).strip()
                    self.match_words[key] = value

        self.model = None
        self.sess = None
        self.g = None

    def load(self):
        if self.is_loaded:
            return

        with tf.device('/device:CPU:0'):
            self.g = tf.Graph()
            with self.g.as_default():
                with tf.variable_scope(self.args.id, reuse=tf.AUTO_REUSE):
                    self.model = Model(self.args, True, self.low_val, self.high_val)
                self.sess = tf.Session()
                tf.global_variables_initializer().run(session=self.sess)
                saver = tf.train.Saver(tf.global_variables(self.args.id))
                ckpt = tf.train.get_checkpoint_state(self.dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.is_loaded = True

    def generate(self, n_tokens, prime, sample, pick, width, quiet, suppress_prime, unk_vocab=None):

        with self.g.as_default():
            return self.model.sample(self.sess, self.words, self.vocab,
                                     n_tokens, prime, sample, pick, width, quiet, suppress_prime, unk_vocab)

    def change_words(self, string):
        words = string.split(" ")
        for i, word in enumerate(words):
            if word in self.match_words:
                words[i] = self.match_words[word]

        return " ".join(words)

    def get_words(self):
        return self.words

    def close(self):
        self.sess.close()
        self.is_loaded = False
