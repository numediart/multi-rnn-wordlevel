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

from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model
from tokenizer import repair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to load stored checkpointed models from')

    parser.add_argument('-n', type=int, default=400,
                        help='number of tokens to sample')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text')

    parser.add_argument('--pick', type=int, default=2,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--count', '-c', type=int, default=20,
                        help='number of samples to print')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing the prime text (default false)')

    args = parser.parse_args()

    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)

    sample(args)


def sample(args):
    config_name = "config"
    index_name = "index"
    preprocess_name = "preprocess"
    with open(os.path.join(args.save_dir, config_name + '.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, os.path.join(preprocess_name, index_name + ".pkl")), 'rb') as f:
        words = cPickle.load(f)
        vocab = dict(zip(words, range(len(words))))

    low_val = next(i for i, word in enumerate(words) if word.startswith("_APPEND_"))
    high_val = len(words) - next(i for i, word in enumerate(reversed(words)) if word.startswith("_APPEND_"))
    # with tf.device('/device:CPU:0'):
    with tf.variable_scope(saved_args.id):
        model = Model(saved_args, True, low_val, high_val)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(args.count):
                print("=== Sample N°" + str(i) + " ===")
                result = model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width,
                                   args.quiet)
                text, _, _ = repair(result)
                print("Dump result : " + result)
                print("Repaired result : " + text)


if __name__ == '__main__':
    main()
