import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filetype='txt', *, name, **kwargs):
        filepath_train_pos = f'{self.directory}{self.train_template}{self.positive_template}'
        filepath_train_neg = f'{self.directory}{self.train_template}{self.negative_template}'
        filepath_test_pos = f'{self.directory}{self.test_template}{self.positive_template}'
        filepath_test_neg = f'{self.directory}{self.test_template}{self.negative_template}'

        train_pos_file_list = os.listdir(filepath_train_pos)
        train_neg_file_list = os.listdir(filepath_train_neg)
        test_pos_file_list = os.listdir(filepath_test_pos)
        test_neg_file_list = os.listdir(filepath_test_neg)

        list = []
        for item in train_neg_file_list:
            path = filepath_train_neg+item
            with open(path, 'r') as fd:
                text = fd.read()

            list.append(text)

        self.data['Train'] = pd.DataFrame(list, columns=['text'])
        self.data['Train']['filename']=train_neg_file_list
        self.data['Train']['POS__NEG']= 0

        list = []
        for item in train_pos_file_list:
            path = filepath_train_pos+item
            with open(path, 'r') as fd:
                text = fd.read()

            list.append(text)

        data = pd.DataFrame(list, columns=['text'])
        data['filename']=train_pos_file_list
        data['POS__NEG']= 1

        self.data['Train'] = pd.concat([self.data['Train'], data])
        self.data['Train'] = self.data['Train'].sample(frac=1).reset_index(drop=True)

        list = []
        for item in test_neg_file_list:
            path = filepath_test_neg+item
            with open(path, 'r') as fd:
                text = fd.read()

            list.append(text)

        self.data['Test'] = pd.DataFrame(list, columns=['text'])
        self.data['Test']['filename']=train_neg_file_list
        self.data['Test']['POS__NEG']= 0

        list = []
        for item in test_pos_file_list:
            path = filepath_test_pos+item
            with open(path, 'r') as fd:
                text = fd.read()

            list.append(text)

        data = pd.DataFrame(list, columns=['text'])
        data['filename']=test_pos_file_list
        data['POS__NEG']= 1

        self.data['Test'] = pd.concat([self.data['Test'], data])
        self.data['Test'] = self.data['Test'].sample(frac=1).reset_index(drop=True)
 #   def save(self, name, filetype='csv', **kwargs):
 #       filepath = f'{self.directory}/{name}.{filetype}'
 #       getattr(self.data[name], f'to_{filetype}')(filepath, **kwargs)
#
 #   def cleanup(self, name, *, drop=False, drop_duplicates=False):
 #       data = self.data[name]
#
 #       if drop is not False:
 #           data = data.drop(columns=drop)
#
 #       if drop_duplicates is True:
 #           data = data.drop_duplicates()
#
 #       self.data['clean'] = data
#
 #   def clean_dataset(self, name):
 #       data = self.data[name]
 #       data.dropna(inplace=True)
 #       indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
 #       self.data['clean'] = data[indices_to_keep].astype(np.float64)
#
 #   def one_hot_encode(self, *, columns):
 #       if 'clean' not in self.data:
 #           print('Can not find clean data')
 #           return
#
 #       data = self.data['clean']
 #       categorical = pd.get_dummies(data[columns], dtype='int')
 #       data = data.drop(data[columns], axis=1)
 #       data = pd.concat([data, categorical], axis=1, sort=False)
 #       self.data['clean'] = data
#
 #       return data
#
 #   def label_encode(self, *, columns):
 #       if 'clean' not in self.data:
 #           print('Can not find clean data')
 #           return
 #       data = self.data['clean']
 #       le = preprocessing.LabelEncoder()
 #       label = pd.DataFrame()
 #       stuff_to_label = columns
 #       ix = 0
 #       for i in stuff_to_label:
 #           le.fit(data[i])
 #           lab = le.transform(data[i])
 #           label.insert(ix, column=i, value=lab)
 #           ix += 1
 #       data = data.drop(stuff_to_label, axis=1)
 #       data = pd.concat([data, label], axis=1, sort=False)
 #       self.data['clean'] = data
#
 #       return data
#
 #   def split_data(self, *, target, size=0.2, state=42):
 #       if 'clean' not in self.data:
 #           print('Can not find clean data')
 #           return
#
 #       data = self.data['clean']
 #       X = np.array(data.drop([target], axis=1))
 #       y = np.array(data[target])
#
 #       X_train, X_test, y_train, y_test = train_test_split(
 #           X, y, test_size=size, random_state=state)
 #       X_train, X_val, y_train, y_val = train_test_split(
 #           X_train, y_train, test_size=size, random_state=state)
#
 #       self.splits = {'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
 #                      'y_train': y_train, 'y_test': y_test, 'y_val': y_val}
#
 #       return X_train, X_test, X_val, y_train, y_test, y_val
#
 #   def scaler(self, *, scale_y=False):
 #       if 'X_train' not in self.splits:
 #           print('Splited data not available')
 #           return
 #       scaler = preprocessing.MinMaxScaler()
 #       self.splits['X_train'] = scaler.fit_transform(self.splits['X_train'])
 #       self.splits['X_test'] = scaler.transform(self.splits['X_test'])
 #       self.splits['X_val'] = scaler.transform(self.splits['X_val'])
#
 #       if scale_y:
 #           self.splits['y_train'] = scaler.fit_transform(
 #               self.splits['y_train'])
 #           self.splits['y_test'] = scaler.transform(self.splits['y_test'])
 #           self.splits['y_val'] = scaler.transform(self.splits['y_val'])

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value
