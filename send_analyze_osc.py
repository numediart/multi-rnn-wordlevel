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

import argparse
import time
import os
from scoring import score_fun
from tokenizer import repair, tokenize
from writer import Writer

from pythonosc import udp_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to load stored checkpointed models from')

    parser.add_argument('--models', nargs="+", type=str, default=['Verne-french-weighted-1024'],
                        help='which models must be used in the save directory')

    parser.add_argument('-n', type=int, default=400,
                        help='number of tokens to sample')
    parser.add_argument('--timeout', type=int, default=60,
                        help='maximum time in seconds before returning a result')
    parser.add_argument('--lazy', default=True, action='store_false',
                        help='writer returns when the score is positive (default true)')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text')
    parser.add_argument('--balance', type=int, default=3,
                        help='Number of samples generated before switching to another model')

    parser.add_argument('--input_encoding', type=str, default='UTF-8',
                        help='character encoding of preprocessed files, from '
                             'https://docs.python.org/3/library/codecs.html#standard-encodings')

    parser.add_argument('--pick', type=int, default=2,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on line returns')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing debug text (default false)')

    parser.add_argument('--osc_host', type=str, default="127.0.0.1", help="IP of the machine listening OSC")
    parser.add_argument('--osc_port', '-p', type=int, default=9000, help="port to send OSC messages")

    check = time.time()
    args = parser.parse_args()

    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)

    client = udp_client.SimpleUDPClient(args.osc_host, args.osc_port)
    print("OSC messages will be sent to {}:{}".format(args.osc_host, args.osc_port))
    client.send_message("/multi_rnn_wordlevel/handshake", 0)

    # Score_fun passed to Writer can also have the following format :
    # {'model_name1':score_fun1, 'model_name2':score_fun2, ...}
    writer = Writer(args.save_dir, args.input_encoding, args.models, args.timeout, args.sample, args.pick, args.width,
                    ['LS', 'DP', 'Hybrid'], score_fun)

    styles = writer.get_styles()
    keep = ""

    args.prime = tokenize(args.prime, config=writer.get_config(styles[0]))

    print("Time: " + str(time.time() - check))
    print()
    check = time.time()

    is_new_sentence = True
    is_left_quote = True

    for i in range(len(styles)):
        print("=== Now using '" + styles[i] + "' ===\n")
        for x in range(args.balance):

            if x == args.balance-1:
                next_style = i+1 if i+1 < len(styles) else 0
                towards_style = styles[next_style]
                print("=== Now style '" + styles[i] + "' shifting towards style '" + towards_style + "' ===")
                print("Usable words percentage: " + str(writer.get_percent_restricted_vocab(styles[i],
                                                                                            towards_style)) + "%\n")
            else:
                towards_style = None

            result = writer.produce(args.prime, args.n, styles[i], False, args.quiet, towards_style)

            send, new_keep = findEOS(result)
            if new_keep.split(" ")[-1] != "\n":
                new_keep += " "

            print("--- Final Output : ")
            output, is_new_sentence, is_left_quote = repair(keep + writer.change_words(send, styles[i]),
                                                is_new_sentence, is_left_quote, config=writer.get_config(styles[i]))

            if client is not None:
                # cut string every 200 characters to avoid OSC packet overflow
                results = [output[j:j + 200] for j in range(0, len(output), 200)]
                j = 1
                for r in results:
                    client.send_message("/multi_rnn_wordlevel/result", (j, len(results), r))
                    j += 1

            print(output)
            print()

            print("--- Kept to group with next : ")
            print(new_keep)
            print()

            keep = writer.change_words(new_keep, styles[i])
            args.prime = result

            print(" --- Time: " + str(time.time() - check))
            print()
            check = time.time()

    writer.stop()


def findEOS(sample):
    line_return = sample.rfind("\n")

    question = sample.rfind("?")
    exclamation = sample.rfind("!")
    dot = sample.rfind(".")

    last_char = max(question, exclamation, dot)

    if last_char > line_return:
        last_pos = last_char
        # Add the double quote if there is any just after.
        if last_pos + 2 < len(sample) and (sample[last_pos + 2] == '"'):
            last_pos = last_pos + 2
    else:
        last_pos = line_return

    # Go just after the very last character
    eos = last_pos + 1

    return sample[:eos], sample[eos:]


if __name__ == '__main__':
    main()
