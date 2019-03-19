import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
import string

class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}
        self.splits = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        directory_template = '{root_dir}/../../data/{name}/'
        self.train_template = 'train/'
        self.test_template = 'test/'
        self.positive_template = 'pos/'
        self.negative_template = 'neg/'
        self.directory = directory_template.format(
            root_dir=root_dir, name=name)
        self.load_data()
        self.dictionary_arr = []
        self.positive_counts, self.negative_counts, self.total_counts = self.count_words('train_raw')
        self.pos_neg_ratios = self.build_count_ratio(self.positive_counts, self.negative_counts, self.total_counts)
        self.dict = self.build_dictionary(self.pos_neg_ratios, 0.000001)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filetype='txt', **kwargs):
        filepath_train_pos = f'{self.directory}{self.train_template}{self.positive_template}'
        filepath_train_neg = f'{self.directory}{self.train_template}{self.negative_template}'
        filepath_test_pos = f'{self.directory}{self.test_template}{self.positive_template}'
        filepath_test_neg = f'{self.directory}{self.test_template}{self.negative_template}'

        train_pos_file_list = os.listdir(filepath_train_pos)
        train_neg_file_list = os.listdir(filepath_train_neg)
        test_pos_file_list = os.listdir(filepath_test_pos)
        test_neg_file_list = os.listdir(filepath_test_neg)

        self.set('train_raw', pd.concat([(self._build_df(filepath_train_neg, train_neg_file_list, 0)),
                                         (self._build_df(filepath_train_pos, train_pos_file_list, 1))]))
        self._shuffle_df('train_raw')
        self.save('train_raw')
        self.set('test_raw', pd.concat([self._build_df(filepath_test_neg, test_neg_file_list, 0),
                                        self._build_df(filepath_test_pos, test_pos_file_list, 1)]))
        self._shuffle_df('test_raw')
        self.save('test_raw')

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value

    def _build_df(self, path, load_list, y_val):
        translator = str.maketrans('', '', string.punctuation)
        list = []
        for item in load_list:
            with open(path+item, 'r') as fd:
                text = fd.read()
                text = text.translate(translator)
            list.append(text)
        data = pd.DataFrame(list, columns=['text'])
        data['POS__NEG'] = y_val
        return data

    def _shuffle_df(self, name):
        self.data[name] = self.data[name].sample(frac=1).reset_index(drop=True)
        return self.data[name]

    def count_words(self, df_name):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()
        touple_word1 = ''
        triple_word1 = ''
        triple_word2 = ''
        for i in range(self.data[df_name].shape[0]):
            if self.data[df_name]['POS__NEG'][i] == 1:
                for word in self.data[df_name]['text'][i].split(" "):
                    touple_word2 = word
                    triple_word3 = word
                    positive_counts[word] += 1
                    positive_counts[touple_word1 + ' ' + touple_word2] += 1
                    positive_counts[triple_word1 + ' ' + triple_word2 + ' ' + triple_word3] += 1

                    total_counts[word] += 1
                    total_counts[touple_word1 + ' ' + touple_word2] += 1
                    total_counts[triple_word1 + ' ' + triple_word2 + ' ' + triple_word3] += 1

                    touple_word1 = touple_word2
                    triple_word1 = triple_word2
                    triple_word2 = triple_word3
            else:
                for word in self.data[df_name]['text'][i].split(" "):
                    touple_word2 = word
                    triple_word3 = word
                    negative_counts[word] += 1
                    negative_counts[touple_word1 + ' ' + touple_word2] += 1
                    negative_counts[triple_word1 + ' ' + triple_word2 + ' ' + triple_word3] += 1

                    total_counts[word] += 1
                    total_counts[touple_word1 + ' ' + touple_word2] += 1
                    total_counts[triple_word1 + ' ' + triple_word2 + ' ' + triple_word3] += 1

                    touple_word1 = touple_word2
                    triple_word1 = triple_word2
                    triple_word2 = triple_word3
        return positive_counts, negative_counts, total_counts

    @classmethod
    def build_count_ratio(cls, count_divident, count_divisor, total_counts):
        ratios = Counter()
        for term, cnt in list(total_counts.most_common()):
            if cnt > 100:
                ratio = count_divident[term] / float(count_divisor[term] + 1)
                ratios[term] = ratio

        return ratios

    @classmethod
    def log_count_ratio(cls, ratios):
        for word, ratio in ratios.most_common():
            ratios[word] = np.log(ratio)
        return ratios

    @classmethod
    def _build_drop_list(cls, ratios, drop_fract):
        drop_count = 0
        drop_words = []

        for word, ratio in ratios.most_common():
            if -drop_fract < ratios[word] < drop_fract:
                drop_words.append(word)
                drop_count += 1
        return drop_words

    @classmethod
    def _reduce_wordcount(cls, ratios, drop_words):
        listx = ratios.copy()
        for dword in drop_words:
            del listx[dword]
        return listx

    @classmethod
    def _set_dict_index(cls, word_list):
        count = 0
        for key, idx in word_list.items():
            word_list[key] = count
            count += 1
        return word_list

    @classmethod
    def build_dictionary(cls, ratios, drop_fract=0.7):
        drop_words = cls._build_drop_list(ratios, drop_fract)
        diction = dict(cls._reduce_wordcount(ratios, drop_words))
        return cls._set_dict_index(diction)

    def build_df(self, name_src, name_dest, dict):
        data = np.zeros((self.data[name_src].shape[0], len(dict)))
        touple_word1 = ''
        triple_word1 = ''
        triple_word2 = ''
        for i in range(self.data[name_src].shape[0]):
            for word in self.data[name_src]['text'][i].split(" "):

                touple_word2 = word
                triple_word3 = word

                if word in dict:
                    data[i][dict[word]] += 1
                if (touple_word1 + ' ' + touple_word2) in dict:
                    data[i][dict[touple_word1 + ' ' + touple_word2]] += 1
                if (triple_word1 + ' ' + triple_word2 + ' ' + triple_word3) in dict:
                    data[i][dict[triple_word1 + ' ' + triple_word2 + ' ' + triple_word3]] += 1

                touple_word1 = touple_word2
                triple_word1 = triple_word2
                triple_word2 = triple_word3
        data = pd.DataFrame(data, columns=dict)
        self.set(name_dest, data)
        self.save(name_dest)
        return data

    def save(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)


