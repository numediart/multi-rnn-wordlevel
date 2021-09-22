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
import os

from tokenizer import repair
from generators import Generator
from analyzers import Analyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--model', type=str, default='Verne-french-weighted-1024',
                        help='which model must be used in the save directory')

    parser.add_argument('-n', type=int, default=400,
                        help='number of tokens to sample')
    parser.add_argument('--count', '-c', type=int, default=10,
                        help='number of samples to print')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text')

    parser.add_argument('--input_encoding', type=str, default='UTF-8',
                        help='character encoding of preprocessed files, from '
                             'https://docs.python.org/3/library/codecs.html#standard-encodings')

    parser.add_argument('--pick', type=int, default=2,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing the prime text (default false)')
    parser.add_argument('--suppress_prime', '-s', default=False, action='store_true',
                        help='suppress the prime text in the returned result (default false))')

    args = parser.parse_args()

    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)

    analyze(args)


def analyze(args):
    analyzer = Analyzer(args.save_dir, args.input_encoding, args.model)

    print_analyze_data(analyzer)

    generator = Generator(args.save_dir, args.input_encoding, args.model)
    generator.load()

    print("=== Tests by sample ===")
    results = []
    for i in range(args.count):
        result = generator.generate(args.n, args.prime, args.sample, args.pick, args.width, args.quiet,
                                    args.suppress_prime)
        results.append(result)
        print_sample(i, generator.change_words(result), analyzer)

        print_stats(analyzer, result)

    print("=== Results for ALL samples ===")
    print_global_stats(analyzer)

    generator.close()


def print_analyze_data(analyzer):
    voc_distrib, data_distrib = analyzer.analyze_data()

    print("--- Vocabulary Distribution ---")
    for corpus in voc_distrib:
        print(corpus + ": " + str(voc_distrib[corpus]) + "%")

    print()

    print("--- Data Distribution ---")
    for corpus in data_distrib:
        print(corpus + ": " + str(data_distrib[corpus]) + "%")

    print()


def print_sample(i, sample, analyzer):
    print("--- Sample N°" + str(i) + " ---")
    print(sample)
    nice_sample, _, _ = repair(sample, config=analyzer.get_config())
    print(nice_sample)
    print()


def print_stats(analyzer, sample):
    stats = analyzer.analyze_sample(sample)

    print_hybridation(stats)
    print_RLS(stats)
    print_LS(stats)
    print_DP(stats)
    print_OC(stats)
    print_VAR(stats)


def print_hybridation(stats):
    if 'Hybrid' not in stats.keys():
        return

    print("--- Corpus Vocabulary Use ---")
    for corpus in stats['Hybrid']['usage']:
        print(corpus + ": " + str(stats['Hybrid']['usage'][corpus]) + "%")
        if corpus != "Common":
            print("\tWords: " + str([word for word in stats['Hybrid']['voc'][corpus]]))

    print()


def print_RLS(stats):
    if 'RLS' not in stats.keys():
        return

    print("--- Original Sequences Detector ---")
    print("Matched values : ")
    print(stats['RLS']['values'])

    print("Diff sequence length between data and sample : ")
    print([stats['RLS']['data_lengths'][i] - stats['RLS']['sample_lengths'][i]
           for i in range(len(stats['RLS']['sample_lengths']))])

    print()


def print_LS(stats):
    if 'LS' not in stats.keys():
        return

    print("--- The longest copied sequence with a tolerance window of " + str(stats['LS']['window']) + " ---")
    print("Length : " + str(stats['LS']['value']) + " (" + str(stats['LS']['percent']) + "% of sample size)")
    print("Sample sequence : " + stats['LS']['sample_seq'].replace("_APPEND_", "_"))  # Help readability
    print("Original sequence : " + stats['LS']['orig_seq'].replace("_APPEND_", "_"))
    print()


def print_DP(stats):
    if 'DP' not in stats.keys():
        return

    print("--- Pattern Detector ---")
    if len(stats['DP']['pattern']) == 0:
        print("No pattern detected.")
    else:
        print("Longest pattern (" + str(len(stats['DP']['pattern'])) + " words) found "
              + str(stats['DP']['occur']) + " times: ")
        print(" ".join(stats['DP']['pattern']))
    print()


def print_OC(stats):
    if 'OC' not in stats.keys():
        return

    print("--- " + str(len(stats['OC']['list'])) + " Most Used Words in samples ---")
    for word, value in stats['OC']['list']:
        if word == '\n':
            print("RETURN LINE: " + str(value) + "%")
        else:
            print(word + " : " + str(value) + "%")
    print("\t> " + str(len(stats['OC']['list'])) + " words represent " +
          str(stats['OC']['total']) + "% of all words in the samples.\n")
    print()


def print_VAR(stats):
    if 'VAR' not in stats.keys():
        return

    print("--- " + str(len(stats['VAR']['list'])) + " Highest Variations against Data Words Usage ---")
    for word, sample_value, data_value in stats['VAR']['list']:
        if word == '\n':
            print("RETURN LINE: " + str(sample_value) + "% against " + str(data_value) + "%")
        else:
            print(word + " : " + str(sample_value) + "% against " + str(data_value) + "%")
    print("\t> Variation Resolution Percentage: " + str(stats['VAR']['resolution']) + "%")
    print()


def print_global_stats(analyzer):
    stats = analyzer.analyze_global()

    if 'Hybrid' in stats.keys():
        print("--- Average Hybridation ---")
        for corpus in stats['Hybrid']:
            print(corpus + ": " + str(stats['Hybrid'][corpus]) + "%")
            if 'Hybrid_Data' in stats.keys():
                print("\twith respect to " + str(stats['Hybrid_Data'][corpus]) + " %")
        print()

    if 'LS' in stats.keys():
        print("--- Longest Sequence in average---")
        print("Length : " + str(stats['LS']['value']) + " (" + str(stats['LS']['percent']) + "% of sample size)")
        print()

    if 'DP' in stats.keys():
        print("--- Average Pattern Detector ---")
        print("Longest pattern (" + str(stats['DP']['length']) + " words) found "
              + str(stats['DP']['occur']) + " times: ")
        print()

    print_VAR(stats)

    print_OC(stats)


if __name__ == '__main__':
    main()
