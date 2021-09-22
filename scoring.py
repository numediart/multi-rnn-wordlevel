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


# Score function should return a number
# If stats are good enough, the number should be above 0 (meaning the result can be returned)


def score_fun(stats):
    plagiat_threshold = 75  # percent
    usage_threshold = 0.5  # percent
    not_so_long_weight = 1
    hybrid_weight = 25

    percent_diff = (stats['LS']['value'] - abs(stats['LS']['diff'])) / stats['nbr_words'] * 100
    not_so_long = -(percent_diff - plagiat_threshold)

    hybrid = 0
    if 'Hybrid' in stats.keys():
        for corpus in stats['Hybrid']['usage']:
            if corpus == "Common":
                continue
            if stats['Hybrid']['usage'][corpus] < usage_threshold:
                hybrid -= 1

    return not_so_long * not_so_long_weight + hybrid * hybrid_weight