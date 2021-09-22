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

import re
import collections
import Stemmer
import random
import os
import json
import codecs
import numpy
import argparse
from six.moves import cPickle

"""
accents: If false, all accents will be removed (default false)
single_line_return: If false, all single line returns will be deleted (default false)
multiple_line_return: If false, all multiple line return will be replaced by only one (default false)
preserve_brackets: If false, all brackets will be replaced by parenthesis (default false)
french_quotes: If set to false, replace opening and closing quotes by casual quotes (default false)
split_compound: If set to true, compound words are split (default true)
long_dash: If set to true, long dash (double dash or solo dash) will be detected and kept (default true)
keep_abbrev: If set to true, keep abbreviation letters together (default true)
keep_proper_nouns: If set to true, it will detect and keep proper nouns with an upper case (default true)
detect_chapters: If set to true, it will search for chapter patterns and remove them (default true)
detect_notes: If set to true, it will search for notes patterns (bracket detection) (default true)
number_classes: If set to true, big numbers will be randomized on units and decimals (default false)
stemming: If set to true, language will be used to split words thanks to stemming algorithm (default true)
language: From ["FR", "EN", "UNK"] used for single quote splitting (default UNK)
special_chars_left_spaced: Add here the special characters you've in your files that should keep a space on left
special_chars_right_spaced: Add here the special characters you've in your files that should keep a space on right
special_chars_no_spaced: Add here the special characters you've in your files that should keep no space around them
special_chars_both_spaced: Add here the special characters you've in your files that should keep spaces around them
special_dict: Add your special replacement here : each key of the dict will replace each single character in the value
special_regex_dict: Add here your special regex : each key of the dict will replace each regex matching value
special_regex_repair_dict: Add here regex to repair special things from tokens to readable output
common_and_proper_words: Fill here words that should remain in upper AND lower case
"""
default_config = {
    "accents": False,
    "single_line_return": False,
    "multiple_line_return": False,
    "preserve_brackets": False,
    "french_quotes": False,
    "split_compound": True,
    "long_dash": True,
    "keep_abbrev": True,
    "keep_proper_nouns": True,
    "detect_chapters": True,
    "detect_notes": True,
    "number_classes": False,
    "stemming": True,
    "language": "UNK",
    "special_chars_left_spaced": "",
    "special_chars_right_spaced": "",
    "special_chars_no_spaced": "",
    "special_chars_both_spaced": "",
    "special_dict": {},
    "special_regex_dict": {},
    "special_regex_repair_dict": {},
    "common_and_proper_words": []
}

accents_dict = {
    "a": "àâä",
    "e": "éèêë",
    "i": "îï",
    "o": "ôö",
    "u": "ùûü",
    "c": "ç",
    "A": "ÂÀÄ",
    "E": "ÉÈÊË",
    "I": "ÎÏ",
    "O": "ÔÖ",
    "U": "ÙÛ",
    "C": "Ç"
}

casual_chars = "()\n,;.!?'\":"

brackets_dict = {
    "(": "[{",
    ")": "]}"
}

quotes_dict = {
   '"': "«»"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        help='base directory containing the data folders')
    parser.add_argument('--subdirs', nargs='+', type=str, default=None,
                        help='which data subdirectories must be used')
    parser.add_argument('--encoding', type=str, default='UTF-8',
                        help='character encoding of the data, from '
                             'https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--output_dir', type=str, default='preprocess',
                        help='directory to store the token files')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='print tokenization operations')

    args = parser.parse_args()

    Tokenizer(args.base_dir, args.subdirs, args.output_dir, args.encoding, args.verbose)

    print("Inputs have been tokenized and saved in : " + args.output_dir)


class Tokenizer:

    def __init__(self, base_dir, subdirs=None, output_dir="preprocess", encoding=None, verbose=False, config_name="config",
                 index_name="index", tensor_name="data"):

        self.verbose = verbose

        self.base_dir = base_dir
        if subdirs is None:
            self.subdirs = os.listdir(base_dir)
        else:
            self.subdirs = subdirs
        self.encoding = encoding
        self.output_dir = output_dir
        self.config_name = config_name
        self.index_name = index_name
        self.tensor_name = tensor_name

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.data = self.compute_data()
        self.index = self.compute_index(self.data)

        with codecs.open(os.path.join(output_dir, self.tensor_name + ".txt"), 'w', encoding=encoding) as f:
                f.write(" ".join(self.data))

        with codecs.open(os.path.join(output_dir, self.index_name + ".txt"), 'w', encoding=encoding) as f:
                f.write("\n".join(self.index))

        words_to_nbr = {x: i for i, x in enumerate(self.index)}
        with open(os.path.join(output_dir, self.index_name + ".pkl"), 'wb') as f:
            cPickle.dump(self.index, f)
        tensor = numpy.array(list(map(words_to_nbr.get, self.data)))
        numpy.save(os.path.join(output_dir, self.tensor_name + ".npy"), tensor)

    def compute_data(self):
        words = []
        for subdir in self.subdirs:
            input_path = os.path.join(self.base_dir, subdir)
            if not os.path.exists(input_path):
                continue

            output_path = os.path.join(self.output_dir, subdir)

            if not os.path.exists(output_path):
                os.mkdir(output_path)

            config_path = os.path.join(input_path, self.config_name + ".json")
            if os.path.exists(config_path):
                with open(config_path) as json_file:
                    config_data = json.load(json_file)

                with open(os.path.join(output_path, self.config_name + ".json"), 'w') as json_output:
                    json.dump(config_data, json_output)
            else:
                config_data = None
                print("No config file found.")

            print("Subdir: " + input_path)
            words += self.compute_subdir(input_path, output_path, config_data)
            words += ['\n']

        return words[:-2]  # Remove last lines return

    def compute_subdir(self, input_path, output_path, config_data):
        data = ""
        for file in os.listdir(input_path):
            path_file = os.path.join(input_path, file)
            output_path_file = os.path.join(output_path, file)

            if os.path.isdir(path_file):
                os.mkdir(output_path_file)
                data += self.compute_subdir(path_file, output_path_file, config_data)

            if os.path.isfile(path_file) and os.path.basename(path_file) != self.config_name + ".json":
                print("File: " + file)
                with codecs.open(path_file, "r", encoding=self.encoding) as f:
                    tmp = tokenize(f.read(), config_data, self.verbose) + " \n "
                    data += tmp

                with codecs.open(output_path_file, 'w', encoding=self.encoding) as f:
                    f.write(tmp)

        words = data.strip().split(" ")

        with codecs.open(os.path.join(output_path, self.index_name + ".txt"), 'w', encoding=self.encoding) as f:
            f.write("\n".join(self.compute_index(words)))

        return words

    def compute_index(self, data):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter()
        word_counts.update(data)

        # Mapping from index to word
        index = [x[0] for x in word_counts.most_common()]
        index = list(sorted(index))

        return index


def tokenize(string, config=None, verbose=False):
    if config is None:
        config = default_config
    else:
        for value in default_config:
            config[value] = default_config[value] if value not in config else config[value]

    string = re.sub(r"\r", "", string)  # Tokenizer support only Unix Format

    for value in config['special_regex_dict']:
        string = re.sub(config['special_regex_dict'][value], value, string)

    string = replace_chars(string, config['special_dict'], verbose=verbose)

    if config['accents']:
        kept_accents_char = "".join([accents_dict[base] for base in accents_dict])
    else:
        kept_accents_char = ""
        string = replace_chars(string, accents_dict, "Replaced vowels", verbose)

    string = remove_notes(string, verbose) if config['detect_notes'] else string

    string = replace_chars(string, brackets_dict, "Replaced brackets", verbose) if not config['preserve_brackets'] \
        else string
    string = replace_chars(string, quotes_dict, "Replaced quotes", verbose) if not config['french_quotes'] else string

    string = re.sub(r"—", "--", string) if config['long_dash'] else string
    kept_regex = re.compile("[^A-Za-z0-9\- " + casual_chars + kept_accents_char + config['special_chars_left_spaced']
                            + config['special_chars_both_spaced'] + config['special_chars_no_spaced']
                            + config['special_chars_right_spaced'] + "]")
    string = remove_unk_chars(string, kept_regex, verbose)  # Clean unauthorized characters

    string = remove_chapters(string, verbose) if config['detect_chapters'] else string

    string = re.sub(r"\n(?!\n)", " ", string) if not config['single_line_return'] else string
    string = re.sub(r"\n{2,}", "\n", string) if not config['multiple_line_return'] else string

    string = re.sub(r"(--)|( - )", " -- ", string) if config['long_dash'] else string
    string = re.sub(r"([^ -])-([^ -])", r"\1 - \2", string) if config['split_compound'] else string

    special_chars = config['special_chars_left_spaced'] + config['special_chars_right_spaced'] \
        + config['special_chars_both_spaced'] + config['special_chars_no_spaced']
    string = re.sub(re.compile("([" + casual_chars + special_chars + "])"), r" \1 ", string)
    string = re.sub(r" {2,}", " ", string)  # Remove redundant whitespaces
    string = re.sub(r"(\. ){2}\.", "...", string)  # Recover '...'

    string = recover_abbrev(string, verbose) if config['keep_abbrev'] else string
    string = clean_shift(string, config['common_and_proper_words'], verbose) if config['keep_proper_nouns'] \
        else string.lower()
    string = recover_quote(string, config['language'])

    string = re.sub(r" {2,}", " ", string).strip()

    string = stem_words(string, config['language']) if config['stemming'] else string

    string = synthesize_numbers(string) if config['number_classes'] else string

    return string


def stem_words(string, language=None):
    if language == 'FR':
        language = "french"
    else:
        language = "english"

    stemmer = Stemmer.Stemmer(language)
    words = string.strip().split(" ")
    stemmed = stemmer.stemWords(words)
    compressed = []
    for i, stem in enumerate(stemmed):
        if len(stem) == 0:
            print(words[i])
        compressed.append(words[i][:len(stem)] + "_FOLLOW_")
        if len(words[i]) != len(stem):
            compressed.append("_APPEND_" + words[i][len(stem):])
    return " ".join(compressed)


def synthesize_numbers(string):
    def replace_num(match):
        string_num = match.group()

        zero_prepend = 0
        while zero_prepend < len(string_num) and string_num[zero_prepend] == '0':
            zero_prepend += 1

        num = int(string_num)
        if num < 10:
            return string_num
        elif num < 100:
            return "[" + string_num[:zero_prepend] + str(int(num/10)) + "X]"
        else:
            return "[" + string_num[:zero_prepend] + str(int(num/100)) + "XX]"

    string = re.sub(r"\b[0-9]+\b", replace_num, string)

    return string


def replace_chars(string, replace_dict, name="Replaced characters", verbose=True):
    correction = 0

    for base in replace_dict:
        string, i = re.subn(re.compile("[" + replace_dict[base] + "]"), base, string)
        correction += i

    if verbose and correction != 0:
        print(name + ": " + str(correction))

    return string


def remove_unk_chars(string, kept_regex, verbose=True):
    avoid = {}

    def avoid_char(match):
        # print(match.start())
        # print(string[match.start()-10:match.start()+10])
        if match.group() in avoid:
            avoid[match.group()] += 1
        else:
            avoid[match.group()] = 1
        return " "

    string = re.sub(kept_regex, avoid_char, string)

    if verbose and len(avoid) != 0:
        # print("Characters have been implicitly blanked.")
        print("Following characters have been implicitly blanked : " + str(avoid))

    return string


def remove_chapters(string, verbose=True):
    def chapter_detected(match):
        if verbose:
            print(match.group().strip())
        return "\n\n"

    chapter = r"\n\s*(CHAP[A-Z.]{0,4} )?((PREMIER)|(FIRST)|[ivxclIVXCL\d]{1,10})\n{1,2}(\s*.+?\n)?\n"
    string = re.sub(chapter, chapter_detected, string)

    return string


def remove_notes(string, verbose=True):
    def notes_detected(match):
        if verbose:
            print(match.group().strip())
        return "\n\n"

    notes = r"(\n\s*\[.+\]( .+(\n.+)*?)?\n\n)"
    string = re.sub(notes, notes_detected, string)

    ref = r"\[\d+\]"
    string = re.sub(ref, notes_detected, string)

    return string


def recover_abbrev(string, verbose=True):
    abbrev_finder = re.compile(r"\b(\w \. ){2,}")
    new_string = list(string)
    shorten = 0
    abbrev_dict = {}
    for find in abbrev_finder.finditer(string):
        new_abbrev = find.group().replace(" ", "") + " "
        new_string[find.start() - shorten:find.start() + len(find.group()) - shorten] = new_abbrev
        shorten += len(find.group()) - len(new_abbrev)
        abbrev_dict[new_abbrev] = 1

    if verbose:
        print("Abbreviations found: " + str([abbrev.strip() for abbrev in abbrev_dict]))

    return "".join(new_string)


def recover_quote(string, language):
    if language == "FR":
        string = re.sub(r"\b([a-zA-Z]) ' ", r"\1' ", string)
        string = re.sub(r"qu ' ", "qu' ", string)
        string = re.sub(r"aujourd ' hui", "aujourd'hui", string)
    elif language == "EN":
        string = re.sub(r" ' (s|S|(?:ve)|(?:VE)|(?:re)|(?:RE)|d|D|(?:ll)|(?:LL)|m|M|[0-9]{2})\b", r" '\1", string)
        string = re.sub(r"n ' t\b", " n't", string)
        string = re.sub(r"N ' T\b", " N'T", string)
        # string = re.sub(r"([^']s) ' ", r"\1 /' ", string)  # Remaining quote can be for plural adjectives

    return string


def clean_shift(string, both_voc, verbose=True):
    lower_both_voc = [x.lower() for x in both_voc]
    proper_voc = detect_shift(string, verbose)

    orig = string
    string = string.lower()

    orig_words = orig.split(" ")
    words = string.split(" ")

    for i, word in enumerate(words):
        if word.lower() in lower_both_voc:
            words[i] = orig_words[i]
        elif word in proper_voc:
            words[i] = proper_voc[word]

    return " ".join(words)


def detect_shift(string, verbose=True):
    global_voc = collections.Counter()
    global_voc.update(string.split(" "))

    proper_finder = re.compile(r"\b[A-Z]\S+ ")

    proper_voc = {}
    for find in proper_finder.finditer(string):
        name = find.group()[:-1]
        index = name.lower()

        # Word doesn't exist in lowercase
        if global_voc[index] == 0:
            if index not in proper_voc:
                proper_voc[index] = name
            elif name.istitle():
                proper_voc[index] = name

    if verbose:
        print("Proper voc found: " + str([proper_voc[index] for index in proper_voc]))

    return proper_voc


def repair(string, is_new_sentence=True, is_left_quote=True, config=None):
    if config is None:
        config = default_config
    else:
        for value in default_config:
            config[value] = default_config[value] if value not in config else config[value]

    casual_chars_left_spaced = "("
    casual_chars_right_spaced = "),;.:"
    casual_chars_no_spaced = "\n'"
    casual_chars_both_spaced = "!?"
    # Double quotes and dashes are repaired here under

    def replace_number_classes(match):
        return match.group()[1:-1].replace('X', str(random.randrange(10)))

    if config['number_classes']:
        string = re.sub(r"\[0*[0-9]+X+\]", replace_number_classes, string)

    if config['stemming']:
        string = re.sub(r" +_APPEND_", "", string)

    string = re.sub(re.compile("([" + casual_chars_left_spaced + casual_chars_no_spaced
                               + config['special_chars_left_spaced'] + config['special_chars_no_spaced']
                               + "]) ([^)" + casual_chars_both_spaced + config['special_chars_both_spaced'] + "])"),
                    r"\1\2", string)

    string = re.sub(re.compile("([^" + casual_chars_both_spaced + config['special_chars_both_spaced'] +
                               "]) ([" + casual_chars_right_spaced + casual_chars_no_spaced
                               + config['special_chars_right_spaced'] + config['special_chars_no_spaced'] + "])"),
                    r"\1\2", string)

    string = re.sub(r" - ", "-", string) if config['split_compound'] else string

    string, is_left_quote = repair_quotes(string, is_left_quote) if not config["french_quotes"] \
        else (string, is_left_quote)

    string, is_new_sentence = repair_shift(string, config, is_new_sentence)

    if config['language'] == "EN":
        string = re.sub(r" n't", "n't", string)
        string = re.sub(r" N'T", "N'T", string)

    for value in config['special_regex_repair_dict']:
        string = re.sub(config['special_regex_repair_dict'][value], value, string)

    return re.sub(r" {2,}", " ", string).strip(" "), is_new_sentence, is_left_quote


def repair_shift(string, config, is_new_sentence=True):
    if config['accents']:
        kept_accents_char = "".join([accents_dict[base] for base in accents_dict])
    else:
        kept_accents_char = ""

    upper_finder = re.compile(r"([.!?]+\s*[\"\']*\s+[\"\']*\s*[a-z" + kept_accents_char + "])|(\n\W*[a-z" +
                              kept_accents_char + "])")
    s = list(string)

    if is_new_sentence:
        first_char_finder = re.compile(r"[a-zA-Z" + kept_accents_char + "]")
        for find in first_char_finder.finditer(string):
            s[find.start()] = s[find.start()].upper()
            break

    for find in upper_finder.finditer(string):
        elem = find.start() + len(find.group()) - 1
        s[elem] = s[elem].upper()

    string = "".join(s)

    # Deal with first character of next sample
    pos = 1
    while string[-pos] in ['"', '\n', ' '] and pos < len(string):
        pos += 1
    if string[-pos] in ['.', '!', '?']:
        is_new_sentence = True
    else:
        is_new_sentence = False

    return string, is_new_sentence


def repair_quotes(string, is_left_quote=True, forced=False):
    # Check last char too
    quote_finder = re.compile(r'"')
    stop_found = re.compile(r'\n+(?!\s*--)').finditer(string)

    new_string = list(string)
    shorten = 0

    # Deal with quotes not ending at line return.
    try:
        last_stop = next(stop_found).start()
    except StopIteration:
        last_stop = len(string)

    for find in quote_finder.finditer(string):
        # Add a quote at the end of the line if quotes are opened
        if not is_left_quote and find.start() > last_stop:
            # The 'minus one' count the whitespace character between end of sentence and the line return
            new_string = new_string[:last_stop - 1 - shorten] + ['"'] + new_string[last_stop - 1 - shorten:]
            shorten -= 1
            is_left_quote = True

        while find.start() > last_stop:
            is_left_quote = True
            try:
                last_stop = next(stop_found).start()
            except StopIteration:
                last_stop = len(string)

        if is_left_quote:
            is_left_quote = False
            if find.start()-shorten+1 < len(new_string) and new_string[find.start() - shorten + 1] == " ":
                new_string[find.start() - shorten: find.start() - shorten + 2] = '"'
                shorten += 1
        else:
            is_left_quote = True
            if find.start() - shorten - 1 >= 0 and new_string[find.start() - shorten - 1] == " ":
                new_string[find.start() - shorten - 1: find.start() - shorten + 1] = '"'
                shorten += 1

    if not is_left_quote and last_stop != len(string):
        # The 'minus one' count the whitespace character between end of sentence and the line return
        new_string = new_string[:last_stop - 1 - shorten] + ['"'] + new_string[last_stop - 1 - shorten:]
        shorten -= 1
        is_left_quote = True

    # Force to close quotes at the end of the sample
    if forced and not is_left_quote:
        i = len(new_string)-1
        while new_string[i] == " ":
            i -= 1

        new_string = new_string[:i+1] + ['"']
        is_left_quote = True

    # Remove useless quotes
    repaired = "".join(new_string).rstrip()
    repaired = re.sub(r'""', '', repaired)
    repaired = re.sub(r'"."', '.', repaired)

    return repaired, is_left_quote


if __name__ == '__main__':
    main()

