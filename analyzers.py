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

import os

import stats
import textloader
import json
from collections import defaultdict


class Analyzers:
    def __init__(self, save_dir, encoding, models=[], stats_used=['RLS', 'LS', 'DP', 'OC', 'VAR', 'Hybrid'],
                 score_functions=None):
        self.analyzers = {}

        if len(models) == 0:
            for model in os.listdir(save_dir):
                if os.path.isdir(os.path.join(save_dir, model)):
                    models.append(model)

        if len(score_functions) != len(models):
            print("ERROR: Number of score functions differs from number of models")
            print("No score function loaded.")
            score_functions = None

        for name in models:
            score_fun = score_functions[name] if score_functions is not None else None
            self.analyzers[name] = Analyzer(save_dir, encoding, name, stats_used, score_fun)

    def analyze(self, name, sample):
        return self.analyzers[name].analyze_sample(sample)

    def score(self, name, sample):
        return self.analyzers[name].score(sample)

    def get_config(self, name):
        return self.analyzers[name].get_config()

    def check_unknown_vocab(self, name, sample):
        return self.analyzers[name].check_unknown_vocab(sample)

    def change_stats(self, name, stats_used):
        self.analyzers[name].change_stats(stats_used)

    def reset(self, name=None):
        if name is None:
            for name in self.analyzers:
                self.analyzers[name].reset_global()
        else:
            self.analyzers[name].reset_global()


class Analyzer:

    def __init__(self, save_dir, encoding, model, stats_used=['RLS', 'LS', 'DP', 'OC', 'VAR', 'Hybrid'],
                 score_fun=None,
                 longest_sequence_window=4,
                 usual_word_occurrence=100, nbr_words_usage=10, config_name="config", preprocess_name="preprocess"):

        self.data = []
        self.corpus_voc = {}
        self.global_stats = dict()
        self.score_fun = score_fun

        self.common_voc = "Common"

        self.longest_sequence_window = longest_sequence_window
        self.usual_word_occurrence = usual_word_occurrence
        self.nbr_words_usage = nbr_words_usage

        self.stats_used = stats_used

        model_dir = os.path.join(save_dir, model)
        corpus_dir = os.path.join(model_dir, preprocess_name)

        self.corpus_list = []
        for name in os.listdir(corpus_dir):
            if os.path.isdir(os.path.join(corpus_dir, name)):
                self.corpus_list.append(name)

        self.is_hybrid = len(self.corpus_list) > 1

        self.reset_global()

        loader = textloader.TextLoader(preprocess_dir=corpus_dir, encoding=encoding)
        for corpus in self.corpus_list:
            temp = loader.load_data([corpus])
            if temp[0] is not '':
                self.data += temp
                self.corpus_voc[corpus] = loader.load_index(temp)

        self.global_voc = loader.load_index(self.data)
        config_path = os.path.join(os.path.join(corpus_dir, self.corpus_list[0]), config_name + ".json")
        if os.path.exists(config_path):
            with open(config_path) as json_file:
                self.config = json.load(json_file)
        else:
            self.config = None

        if self.is_hybrid:
            self.merged_vocabs = stats.merge_common_vocab(self.corpus_voc, self.common_voc)

        self.data_index = defaultdict(list)
        for index, word in enumerate(self.data):
            self.data_index[word].append(index)

        self.rev_data = list(reversed(self.data))

    def change_stats(self, stats_used):
        self.stats_used = stats_used
        self.reset_global()

    def check_unknown_vocab(self, sample):
        words = sample.split(" ")
        unk = []
        for word in words:
            if word is not '' and word not in self.global_voc:
                unk.append(word)
        return unk

    def get_config(self):
        return self.config

    def analyze_data(self):
        if self.is_hybrid and 'Hybrid' in self.stats_used:
            self.global_stats['Hybrid_Data'] = dict()
            voc_distrib, _ = stats.vocab_distribution(self.global_voc, self.merged_vocabs)
            data_distrib, _ = stats.vocab_distribution(self.data, self.merged_vocabs)

            for corpus in self.merged_vocabs:
                self.global_stats['Hybrid_Data'][corpus] = data_distrib[corpus]

            return voc_distrib, data_distrib

        return [], []

    def analyze_sample(self, sample):
        sample_stats = dict()

        words = sample.split(" ")
        sample_stats['nbr_words'] = len(words)

        self.global_stats['nbr'] += 1
        self.global_stats['words'] += words

        if self.is_hybrid and 'Hybrid' in self.stats_used:
            sample_stats['Hybrid'] = dict()
            sample_stats['Hybrid']['usage'], sample_stats['Hybrid']['voc'] = stats.vocab_distribution(words,
                                                                                                      self.merged_vocabs
                                                                                                      )
            for corpus in self.merged_vocabs:
                self.global_stats['Hybrid'][corpus] += sample_stats['Hybrid']['usage'][corpus]

        if 'RLS' in self.stats_used:
            sample_stats['RLS'] = dict()
            sample_stats['RLS']['values'], sample_stats['RLS']['sample_lengths'], sample_stats['RLS'][
                'data_lengths'] = stats.recurse_longest_sequence(words, self.data, self.rev_data, self.data_index)

        if 'LS' in self.stats_used:
            sample_stats['LS'] = dict()
            sample_stats['LS']['value'], sample_stats['LS']['percent'], sample_stats['LS']['sample_seq'], _, \
            sample_stats['LS']['orig_seq'] = stats.longest_original_sequence(words, self.data, self.rev_data,
                                                                             self.data_index,
                                                                             self.longest_sequence_window,
                                                                             self.usual_word_occurrence)
            sample_stats['LS']['diff'] = len(sample_stats['LS']['sample_seq']) - len(sample_stats['LS']['orig_seq'])
            sample_stats['LS']['window'] = self.longest_sequence_window
            self.global_stats['LS']['value'] += sample_stats['LS']['value']
            self.global_stats['LS']['diff'] += sample_stats['LS']['diff']

        if 'DP' in self.stats_used:
            sample_stats['DP'] = dict()
            sample_stats['DP']['pattern'], sample_stats['DP']['occur'] = stats.detect_pattern(words)

            self.global_stats['DP']['length'] += len(sample_stats['DP']['pattern'])
            self.global_stats['DP']['occur'] += sample_stats['DP']['occur']

        if 'OC' in self.stats_used:
            sample_stats['OC'] = dict()
            sample_stats['OC']['total'], sample_stats['OC']['list'] = stats.occurrences(words, self.nbr_words_usage)

        if 'VAR' in self.stats_used:
            sample_stats['VAR'] = dict()
            sample_stats['VAR']['resolution'], sample_stats['VAR']['list'] = stats.variation(words, self.data,
                                                                                             self.nbr_words_usage)

        return sample_stats

    def analyze_global(self):
        all_stats = dict()

        if 'LS' in self.stats_used:
            all_stats['LS'] = dict()
            all_stats['LS']['value'] = self.global_stats['LS']['value'] / self.global_stats['nbr']
            words_per_sample = len(self.global_stats['words']) / self.global_stats['nbr']
            all_stats['LS']['percent'] = all_stats['LS']['value'] / words_per_sample * 100
            all_stats['LS']['diff'] = self.global_stats['LS']['diff'] / self.global_stats['nbr']

        if 'DP' in self.stats_used:
            all_stats['DP'] = dict()
            all_stats['DP']['length'] = self.global_stats['DP']['length'] / self.global_stats['nbr']
            all_stats['DP']['occur'] = self.global_stats['DP']['occur'] / self.global_stats['nbr']

        if self.is_hybrid and 'Hybrid' in self.stats_used:
            all_stats['Hybrid'] = dict()
            for corpus in self.global_stats['Hybrid']:
                all_stats['Hybrid'][corpus] = self.global_stats['Hybrid'][corpus] / self.global_stats['nbr']
            all_stats['Hybrid_Data'] = self.global_stats['Hybrid_Data']

        if 'OC' in self.stats_used:
            all_stats['OC'] = dict()
            all_stats['OC']['total'], all_stats['OC']['list'] = stats.occurrences(self.global_stats['words'],
                                                                                  self.nbr_words_usage)
        if 'VAR' in self.stats_used:
            all_stats['VAR'] = dict()
            all_stats['VAR']['resolution'], all_stats['VAR']['list'] = stats.variation(self.global_stats['words'], self.data,
                                                                                  self.nbr_words_usage)

        return all_stats

    def reset_global(self):
        self.global_stats['nbr'] = 0
        self.global_stats['words'] = []

        self.global_stats['LS'] = dict()
        self.global_stats['LS']['value'] = 0
        self.global_stats['LS']['diff'] = 0

        self.global_stats['DP'] = dict()
        self.global_stats['DP']['length'] = 0
        self.global_stats['DP']['occur'] = 0

        if self.is_hybrid:
            self.global_stats['Hybrid'] = dict()
            self.global_stats['Hybrid'][self.common_voc] = 0
            for corpus in self.corpus_list:
                self.global_stats['Hybrid'][corpus] = 0

    def score(self, sample):
        if self.score_fun is None:
            return 0
        else:
            return self.score_fun(self.analyze_sample(sample))