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

import time
import random
from generators import Generators
from analyzers import Analyzers


class Writer:

    def __init__(self, save_dir, encoding, models, timeout, sample, pick, width, stats_list, score_fun):
        self.timeout = timeout

        if isinstance(score_fun, dict):
            score_functions = score_fun
        else:
            score_functions = dict()
            for model in models:
                score_functions[model] = score_fun

        self.gen = Generators(save_dir, encoding, models, len(models))
        self.ana = Analyzers(save_dir, encoding, models, stats_list, score_functions)
        self.sample = sample
        self.pick = pick
        self.width = width

    def produce(self, base_prime, length, style, try_hard=True, quiet=False, towards_style=None):
        best_result = ""
        best_score = -1e10
        check = time.time()
        search = True

        unk = self.ana.check_unknown_vocab(style, base_prime)
        if not quiet and len(unk) != 0:
            print("--- Unknown words in base prime : " + " ".join(unk))
            print()

        while search:
            # If we search again, we need to add some randomness in the prime
            prime, len_prime = Writer.randomize_prime(base_prime)
            n_tokens = length + len_prime

            # Quiet must be set to True here or the prime will be in the result too
            # This will lead to unexpected outputs and unexpected stats
            result = self.gen.generate(style, n_tokens, prime, self.sample, self.pick, self.width, True,
                                       True, towards_style)
            score = self.ana.score(style, result)

            if score >= best_score:
                best_score = score
                best_result = result

            if time.time() - check >= self.timeout:
                search = False
            elif not try_hard:
                search = best_score <= 0

            if not quiet:
                # Prime is a random token chosen by model
                if prime == "":
                    prime = result.split(" ")[0]

                print("--- Random Prime : " + prime)
                print()
                print("--- DEBUG : " + result)
                print("--- Score : " + str(score))
                print()

        return best_result

    def get_styles(self):
        return self.gen.get_names()

    def get_config(self, style):
        return self.ana.get_config(style)

    def get_percent_restricted_vocab(self, style1, style2):
        return self.gen.get_percent_restricted_vocab(style1, style2)

    @staticmethod
    def randomize_prime(base_prime):
        words = base_prime.split(" ")
        if len(words) != 0:
            start = random.randrange(len(words))
            prime = " ".join(words[start:])
            return prime, len(words) - start

        return "", 0

    def change_words(self, string, style):
        return self.gen.change_words(style, string)

    def stop(self):
        self.gen.close()
