# coding: utf-8
# Copyright (c) pytmge Development Team.

"""
Extracting features based on electron orbital attributes.

"""


import os
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from pytmge.core import elemental_data
from pytmge.core.crystal import extract_composition
from pytmge.core.plugins import progressbar


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


class feature_design:
    '''
    Extracting features based on electron orbital attributes.

    '''

    def __init__(self):
        self._elemental_attribute_format = '[attribute].[shell_selection].[math operator 1]'
        self._feature_format = '[attribute].[shell_selection].[math operator 1].[math operator 2]'

    @staticmethod
    def delete_unusable_features(df_features):
        '''
        Delete the features having empty value(s)
        and the features being of zero variance.

        Parameters
        ----------
        df_features : DataFrame
            features.

        Returns
        -------
        df_usable_features : DataFrame
            usable_features.

        '''

        print('deleting unusable features')
        features_variance = np.var(df_features, axis=0)
        df_usable_features = df_features.loc[:, features_variance != 0] * 1

        # df_usable_features.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'usable_feature_variables.csv')

        print(df_usable_features.shape[0], 'entries', df_usable_features.shape[1], 'usable features')

        return df_usable_features

    @classmethod
    def get_features(self, df_dataset):
        '''
        Extracting features.

        Parameters
        ----------
        df_dataset : DataFrame
            chemical formulas as index.

        Returns
        -------
        df_features : DataFrame
            features.

        '''

        df_composition = extract_composition(df_dataset)
        df_orbital_attributes_of_elements = elemental_data.orbital_attributes_of_elements

        # # Lite edition
        # _ea = df_orbital_attributes_of_elements.loc[
        #     [
        #         'E' in col_name
        #         and 'range' in col_name
        #         for col_name in list(df_orbital_attributes_of_elements.index)
        #     ], :
        # ]
        # df_orbital_attributes_of_elements = _ea * 1
        # #

        print(df_orbital_attributes_of_elements.shape[0], 'attributes,', df_composition.shape[0], 'entries.')

        cache_path = str(Path(__file__).absolute().parent) + '\\' + '_cache\\'
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.makedirs(cache_path)

        chemical_formula_list = list(df_composition.index)
        elements_existence = (df_composition.notnull() * 1).replace(0, np.nan)

        math_operators = {
            'sum': 'sum(x)',
            'avg': 'sum(x)/N',
            'wavg': 'sum(w*x)/sum(w)',
            'max': 'max(x)',
            'min': 'min(x)',
            'range': 'max(x)-min(x)',
            'std': '(sum((x-wavg)**2)/N)**(1/2)'
        }

        warnings.filterwarnings('ignore')

        n = 0
        for a in list(df_orbital_attributes_of_elements.index):

            ea = df_orbital_attributes_of_elements.loc[a, :] * 1
            features = {}
            for o in math_operators.keys():
                features[a + '.' + o] = {}
            if ea.notnull().sum() == 0:
                # when all atomic_attribute values are nan, their feature_variables should be nan, too.
                for o in math_operators.keys():
                    for cf in chemical_formula_list:
                        features[a + '.' + o][cf] = np.nan
            else:
                for cf in chemical_formula_list:
                    v = ea * elements_existence.loc[cf]
                    w = df_composition.loc[cf]  # weightings
                    wea = v * w

                    # compound_wavg is nan if this atomic_attribute are_empty for all elements.
                    # compound_wavg = [np.nansum(wea) / w.sum(), np.nan][wea.notnull().sum() == 0]
                    compound_wavg = np.nansum(wea) / w.sum() if wea.notnull().sum() else np.nan

                    # compound_sum is nan if this atomic_attribute are_empty for all elements.
                    # compound_sum = [np.nansum(v), np.nan][v.notnull().sum() == 0]
                    compound_sum = np.nansum(v) if v.notnull().sum() else np.nan

                    features[a + '.max'][cf] = np.nanmax(v)
                    features[a + '.min'][cf] = np.nanmin(v)
                    features[a + '.range'][cf] = np.nanmax(v) - np.nanmin(v)
                    features[a + '.std'][cf] = np.nanstd(v)
                    features[a + '.avg'][cf] = np.nanmean(v)
                    features[a + '.wavg'][cf] = compound_wavg
                    features[a + '.sum'][cf] = compound_sum

                    n += 1
                    progressbar(n, df_composition.shape[0] * df_orbital_attributes_of_elements.shape[0])

            _df_features = pd.DataFrame.from_dict(features, orient='columns', dtype='float64')
            _df_features.to_csv(cache_path + 'feature_variables [' + a + '._].csv', float_format='%8f')

        warnings.resetwarnings

        # reload features
        df_features = pd.DataFrame(dtype='float64')
        file_list = os.listdir(cache_path)
        n = 0
        for i in file_list:
            n += 1
            progressbar(n, len(file_list))
            if os.path.isfile(cache_path + i):
                df_data = pd.read_csv(cache_path + i, index_col=0)
                for feature_name in list(df_data.columns):
                    df_features[feature_name] = df_data[feature_name] * 1

        # df_features.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'feature_variables.csv', float_format='%8f')

        shutil.rmtree(cache_path)

        return df_features


#

class feature_engineering:

    @staticmethod
    def get_correlation_matrix(df_features):
        '''
        Getting Pearson correlation matrix.

        Parameters
        ----------
        df_features : DataFrame
            df_features.

        Returns
        -------
        df_correlation_matrix : DataFrame
            df_correlation_matrix.

        '''

        df_correlation_matrix = pd.DataFrame(np.corrcoef(df_features, rowvar=0))
        df_correlation_matrix.index = df_features.columns
        df_correlation_matrix.columns = df_features.columns

        return df_correlation_matrix

    @staticmethod
    def feature_selection_by_Pearson_correlation(df_features):
        feature_subset = []
        return feature_subset
