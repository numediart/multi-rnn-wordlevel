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

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np

from beam import BeamSearch


class Model:
    def __init__(self, args, infer=False, start_range_weight=0, end_range_weight=0):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                # with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                # tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                # tf.summary.histogram('histogram', var)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        labels = tf.reshape(self.targets, [-1])
        logical_append = tf.logical_and(tf.greater_equal(labels, start_range_weight), tf.less(labels, end_range_weight))
        weights = tf.multiply(4.0, tf.cast(logical_append, tf.float32)) + 1
        # [tf.ones([args.batch_size * args.seq_length])]

        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                       [labels],
                                                       [weights],
                                                       args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4, quiet=False,
               suppress_prime=False, unk_vocab=None):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime_labels, width):
            """Returns the beam search pick."""
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)

            if unk_vocab is not None:
                unk_labels = [vocab.get(word) for word in unk_vocab]
            else:
                unk_labels = None

            samples, scores = bs.search(unk_labels, None, k=width, maxsample=num, use_unk=(unk_vocab is not None))

            return samples[np.argmin(scores)]

        ret = ''
        if not len(prime) or prime == ' ':
            prime = random.choice(list(vocab.keys()))
            # If we choose prime here, we need to return it
            suppress_prime = False

        if not quiet:
            print(prime)

        if not suppress_prime:
            ret = prime

        # TODO Give better than the 0-index word if word doesn't exist
        prime_labels = [vocab.get(word, 0) for word in prime.split(" ")]

        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            for label in prime_labels[:-1]:
                x = np.zeros((1, 1))
                x[0, 0] = label
                feed = {self.input_data: x, self.initial_state: state}
                [state] = sess.run([self.final_state], feed)

            sample = prime_labels[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = sample
                feed = {self.input_data: x, self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if words[sample] == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else:  # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = words[sample]
                ret += ' ' + pred
        elif pick == 2:
            pred = beam_search_pick(prime_labels, width)
            for i, label in enumerate(pred[len(prime_labels):]):
                ret += ' ' + words[label]

        return ret
