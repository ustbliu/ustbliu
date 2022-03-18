# coding: utf-8
# Copyright (c) pytmge Development Team.

'''
Classes for data preparation.

'''


import numpy as np
import pandas as pd

from pytmge.core.crystal import extract_composition
from pytmge.core.plugins import progressbar


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


class data_preparation:

    def __init__(self):
        pass

    @staticmethod
    def delete_duplicates(df_dataset):
        '''
        Delete duplicate entries in dataset.

        If two or more entries have the same chemical formula
        but different property values, keep the entry having greater value (of the first column).

        Parameters
        ----------
        df_dataset : DataFrame
            dataset of chemical formula and target variable.
            chemical formulas as index.

        Returns
        -------
        deduped_subset : DataFrame
            deduped subset.

        '''

        print('\ndeleting duplicate entries in dataset ...')

        # sort the dataset by the values of first column.
        for col_name in list(df_dataset.columns)[::-1]:
            df_dataset = df_dataset.sort_values(by=col_name)

        deduped_subset = {}
        for i, cf in enumerate(list(df_dataset.index)):
            deduped_subset[cf] = df_dataset.loc[cf]
            progressbar(i + 1, df_dataset.shape[0])

        print('  original:', i + 1, '| deduped:', len(deduped_subset.keys()))

        df_deduped_subset = pd.DataFrame.from_dict(deduped_subset, orient='index')
        df_deduped_subset.sort_index(ascending=True, inplace=True)
        # df_deduped_subset.sort_values(
        #     by=list(df_deduped_subset.columns)[0],
        #     ascending=False,
        #     inplace=True
        # )

        return df_deduped_subset

    @classmethod
    def categorization_by_composition(cls, df_dataset):
        '''
        Categorizing the chemical formulas,
        according to 'number_of_elements', 'element', and 'elemental_contents' (n-e-c).

        Parameters
        ----------
        df_dataset : DataFrame
            dataset of chemical formula and target variable.
            chemical formulas as index.

        Returns
        -------
        dict_category : dict
            dict_category.

        '''

        print('\ncategorizing the chemical formulas ...')

        df_composition = extract_composition(df_dataset)
        cfs = list(df_composition.index)
        _elements = list(df_composition.columns)

        # elemental contents
        contents = df_composition.fillna(0).applymap(lambda x: np.int(x + 0.5))

        # number of elements in each chemical formula, ignore the element(s) that content < 0.5
        number_of_elements = (df_composition.applymap(lambda x: x >= 0.5) * 1).sum(axis=1)

        # print('labeling ...')
        labels = {}
        for cf in cfs:
            labels[cf] = {}
            i = 0
            for e in _elements:
                if contents.loc[cf, e] >= 0.5:
                    i += 1
                    # assign a category lable 'n-e-c' to each chemical formula
                    n = str(round(number_of_elements[cf]))
                    c = str(int(contents.loc[cf, e] + 0.5))
                    labels[cf][i] = n + '-' + e + '-' + c

        # print('categorizing ...')
        dict_category = {}
        for cf in cfs:
            for label in labels[cf].values():
                dict_category[label] = {}
        for cf in cfs:
            for label in labels[cf].values():
                dict_category[label][cf] = df_dataset.loc[cf, :].to_dict()

        return dict_category

    @classmethod
    def subset(cls, df_dataset):
        '''
        Getting subset.
        For each category, pick one entry having the highest value of material property.

        Parameters
        ----------
        df_dataset : DataFrame
            dataset of chemical formula and target variable.
            chemical formulas as index.

        Returns
        -------
        df_subset : DataFrame
            df_subset.

        '''

        dict_category = cls.categorization_by_composition(df_dataset)
        dict_subset = {}
        for category_label, dict_entries in dict_category.items():
            df_entries = pd.DataFrame.from_dict(dict_entries, orient='index')
            df_highest = df_entries.sort_values(by=df_entries.columns[0], ascending=False)
            ds_highest = df_highest.iloc[0, :]
            dict_subset[ds_highest.name] = ds_highest.to_dict()
        df_subset = pd.DataFrame.from_dict(dict_subset, orient='index')
        return df_subset

    @staticmethod
    def distances_within_dataset(df_dataset):
        '''
        Manhattan distances in composition space,
        between each two chemical formulas in the dataset.

        Parameters
        ----------
        df_dataset : DataFrame
            dataset of chemical formula and target variable.
            chemical formulas as index.

        Returns
        -------
        df_distances : DataFrame
            df_distances.
            chemical formulas as index and columns.

        '''

        df_composition = extract_composition(df_dataset)
        cfs = list(df_composition.index)
        df_distances = pd.DataFrame(index=cfs, columns=cfs, dtype='float')
        c = df_composition.fillna(0)

        for i, cf in enumerate(cfs):
            d = np.abs(c - c.loc[cf, :])
            df_distances[cf] = np.nansum(d, axis=1)

            progressbar(i + 1, len(cfs))

        return df_distances

    @staticmethod
    def distances_between_datasets(df_dataset_0, df_dataset_1):
        '''
        Manhattan distances in composition space,
        from each point in dataset_0 to the points in dataset_1.

        Parameters
        ----------
        df_dataset_0 : DataFrame
            df_composition, derived from class 'dataset' .
            chemical formulas as index, elements as columns.

        df_dataset_1 : DataFrame
            df_composition, derived from class 'dataset' .
            chemical formulas as index, elements as columns.

        Returns
        -------
        df_distances : DataFrame
            df_distances.
            chemical formulas in dataset_0 as index.

        '''

        df_composition_0 = extract_composition(df_dataset_0)
        df_composition_1 = extract_composition(df_dataset_1)
        c0 = df_composition_0.fillna(0)
        c1 = df_composition_1.fillna(0)
        cfs0 = list(df_composition_0.index)
        cfs1 = list(df_composition_1.index)

        df_distances = pd.DataFrame(index=cfs0, columns=cfs1, dtype='float')

        for i, cf in enumerate(cfs1):
            _d = np.abs(c0 - c1.loc[cf, :])
            df_distances[cf] = np.nansum(_d, axis=1)

            progressbar(i + 1, len(cfs1))

        return df_distances
