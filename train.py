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
import time
import os
from six.moves import cPickle
import uuid

import textloader
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_encoding', type=str, default='UTF-8',
                        help='character encoding of input.txt, from '
                             'https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save/test',
                        help='directory to store the model')

    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.666,
                        help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--gpu_nbr', type=int, default=0,
                        help='which gpu device should be used')
    parser.add_argument('--gpu_mem_growth', default=False, action='store_true',
                        help='rather than fixing memory, authorize gpu memory usage to grow')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()

    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    if args.init_from is not None:
        args.init_from = os.path.join(os.path.dirname(__file__), args.init_from)

    model_id = uuid.uuid4().hex
    args.id = model_id

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train(args)


def train(args):
    preprocess_name = "preprocess"
    config_name = "config"
    index_name = "index"
    preprocess_dir = os.path.join(args.save_dir, preprocess_name)
    loader = textloader.TextLoader(args.batch_size, args.seq_length, preprocess_dir,
        encoding=args.input_encoding)

    print("Loading preprocessed files from " + preprocess_dir + "...")
    loader.load_preprocessed()
    loader.create_batches()
    loader.reset_batch_pointer()

    args.vocab_size = loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a path" % args.init_from
        assert os.path.isfile(
            os.path.join(args.init_from, config_name + ".pkl")), "config pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, os.path.join(preprocess_name,
                                           index_name + ".pkl"))), "index pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, config_name + '.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[
                checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, os.path.join(preprocess_name, index_name + '.pkl')), 'rb') as f:
            saved_words = cPickle.load(f)
            saved_vocab = dict(zip(saved_words, range(len(saved_words))))
        assert saved_words == loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab == loader.vocab, "Data and loaded model disagree on dictionary mappings!"

        args.id = saved_model_args.id

    low_val = next(i for i, word in enumerate(loader.words) if word.startswith("_APPEND_"))
    high_val = len(loader.words) - next(i for i, word in enumerate(reversed(loader.words)) if word.startswith("_APPEND_"))

    with open(os.path.join(args.save_dir, config_name + '.pkl'), 'wb') as f:
        cPickle.dump(args, f)

    device_nbr = str(args.gpu_nbr)
    with tf.device('/device:GPU:'+device_nbr):
        with tf.variable_scope(args.id, reuse=tf.AUTO_REUSE):
            model = Model(args, False, low_val, high_val)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    if args.gpu_mem_growth:
        gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=device_nbr)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem, visible_device_list=device_nbr)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            for b in range(loader.pointer, loader.num_batches):
                start = time.time()
                x, y = loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                train_writer.add_summary(summary, e * loader.num_batches + b)
                speed = time.time() - start
                if (e * loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                          .format(e * loader.num_batches + b,
                                  args.num_epochs * loader.num_batches,
                                  e, train_loss, speed))
                if (e * loader.num_batches + b) % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and b == loader.num_batches - 1):  # save for the last result
                    with open(os.path.join(args.save_dir, 'logs.txt'), 'a+') as the_file:
                        the_file.write("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f} \n"
                                       .format(e * loader.num_batches + b,
                                               args.num_epochs * loader.num_batches,
                                               e, train_loss, speed))
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        train_writer.close()


if __name__ == '__main__':
    main()
