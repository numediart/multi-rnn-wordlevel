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

from collections import Counter

def occurrences(words, nbr_words=None):
    word_counts = Counter()
    word_counts.update(words)

    top_occurrences = []
    total = 0
    for x in word_counts.most_common(nbr_words):
        percentage = x[1] / len(words) * 100
        top_occurrences.append((x[0], percentage))
        total += percentage

    return total, top_occurrences


def variation(words, data, nbr_words=None):
    word_counts = Counter()
    word_counts.update(words)

    data_counts = Counter()
    data_counts.update(data)

    for x in data_counts.most_common():
        data_counts[x[0]] = x[1] / len(data) * 100

    diff = Counter()
    for x in word_counts.most_common():
        word_counts[x[0]] = x[1] / len(words) * 100
        diff[x[0]] = abs(word_counts[x[0]] - data_counts[x[0]])

    resolution = 100 / len(words)

    variations = []
    for word, _ in diff.most_common(nbr_words):
        variations.append((word, word_counts[word], data_counts[word]))

    return resolution, variations


def recurse_longest_sequence(words, data, rev_data, data_index, window=4, usual_word_occurrence=100):
    if len(words) > 0:
        value, _, sample_seq, index, orig_seq = longest_original_sequence(words, data, rev_data, data_index, window,
                                                                          usual_word_occurrence)

        if sample_seq != '':
            size = len(sample_seq.split(" "))

            prev_values, prev_len_sample, prev_len_data = recurse_longest_sequence(words[:index], data, rev_data, data_index, window,
                                                                                   usual_word_occurrence)
            post_values, post_len_sample, post_len_data = recurse_longest_sequence(words[index + size:], data, rev_data, data_index, window,
                                                                                   usual_word_occurrence)

            return prev_values + [value] + post_values, prev_len_sample + [size] + post_len_sample, \
                   prev_len_data + [len(orig_seq.split(" "))] + post_len_data

    return [], [], []


def longest_original_sequence(words, data, rev_data, data_index, window=4, usual_word_occurrence=100):
    # sample and data are array with each entry corresponding to a word in the vocabulary
    max_value = 0
    max_words_index = 0
    max_data_index = 0
    max_words_len = 0
    max_data_len = 0

    rev_words = list(reversed(words))

    for i, word in enumerate(words):
        indices = data_index[word]

        if len(indices) > usual_word_occurrence:
            # Discard starting algorithm on usual words
            # They will not help us and will improve computing time
            continue

        for index in indices:
            rev_value, rev_words_len, rev_data_len = match_sequences(rev_words[len(rev_words)-i:],
                                                                     rev_data[len(rev_data)-i:],
                                                                     window)

            value, words_len, data_len = match_sequences(words[i:],
                                                         data[index:],
                                                         window)
            if rev_value + value > max_value:
                max_value = rev_value + value
                max_words_index = i - rev_words_len
                max_data_index = index - rev_data_len
                max_words_len = words_len + rev_words_len
                max_data_len = data_len + rev_data_len

    percent = max_value / len(words) * 100
    sample_seq = ' '.join(words[max_words_index:max_words_index + max_words_len])
    orig_seq = ' '.join(data[max_data_index:max_data_index + max_data_len])

    return max_value, percent, sample_seq, max_words_index, orig_seq


def match_sequences(words, data, window):
    # First element in words and data are matching
    # Return the size of the longest sequence found with a flexibility window

    if len(words) == 0:
        return 0, 0, 0

    seq = 1
    data_pointer = 1
    words_pointer = 1
    while data_pointer < len(data) and words_pointer < len(words):
        real_window = min(len(data) - data_pointer, len(words) - words_pointer, window)

        words_window = words[words_pointer:words_pointer + real_window]
        data_window = data[data_pointer:data_pointer + real_window]

        found, i, j = match_window(words_window, data_window)

        if not found:
            break

        seq += 1
        words_pointer += i + 1
        data_pointer += j + 1

    return seq, words_pointer, data_pointer


def match_window(data_1, data_2):
    # Find index of first corresponding element
    for i in range(len(data_1)):
        for j in range(len(data_2)):
            if data_1[i] == data_2[j]:
                return True, i, j
    return False, -1, -1


def vocab_distribution(words, vocabs):
    use = {}
    sample_vocabs = {}
    for name in vocabs:
        use[name] = 0
        sample_vocabs[name] = {}

    for word in words:
        for name in vocabs:
            if word in vocabs[name].keys():
                use[name] += 1
                sample_vocabs[name][word] = 1

    for name in vocabs:
        use[name] = use[name] / len(words) * 100

    return use, sample_vocabs


def merge_common_vocab(vocabs, common_name='Common'):
    new_vocabs = {}
    for name in vocabs:
        new_vocabs[name] = vocabs[name].copy()
        if name == common_name:
            print("Error : Common name is also a vocab name.")
            return None

    new_vocabs[common_name] = {}

    # Iterate over words of one random vocab
    ref = dict(next(iter(vocabs.values())))
    for word in ref:
        common = True
        for name in vocabs:
            if word not in vocabs[name]:
                common = False

        if common:
            new_vocabs[common_name][word] = 1
            for name in vocabs:
                new_vocabs[name].pop(word)

    return new_vocabs


def detect_pattern(words, max_size=None):
    best_pattern = None

    # Don't consider these tokens in patterns detection
    escape_words = ["\n", '"', ".", "'", ",", ";"]

    for word in escape_words:
        words = [x for x in words if x != word]

    if max_size is None or max_size > int(len(words) / 2):
        max_size = int(len(words) / 2)

    # Checking 1 word for patterns is not very interesting
    # counter = Counter()
    # counter.update(words)
    # result = counter.most_common(1)[0]
    # best_pattern = ([result[0]], result[1])

    for size in range(2, max_size + 1):
        counter = {}
        grep = (words[i:] for i in range(size))
        patterns = list(zip(*grep))

        # Avoid overlapping patterns
        for i, pattern in enumerate(patterns):

            if pattern not in counter:
                counter[pattern] = {'count': 1, 'index': i}

            elif (pattern in counter) and counter[pattern]['index'] + size <= i:
                counter[pattern]['count'] += 1
                counter[pattern]['index'] = i

        result = None
        for pattern in counter:
            if result is None or result[1] < counter[pattern]['count']:
                result = (pattern, counter[pattern]['count'])

        if result[1] > 1 and (
                best_pattern is None or result[1] * len(result[0]) > best_pattern[1] * len(best_pattern[0])):
            best_pattern = result

    if best_pattern is None:
        return [], 0

    return best_pattern
