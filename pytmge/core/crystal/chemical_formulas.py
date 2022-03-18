# coding: utf-8
# Copyright (c) pytmge Development Team.

'''
Dealing with chemical formulas.

The format of the chemical formulas is supposed to be like 'H2O1' or 'C60',
whereas 'H2O' or 'C' is not ok.
Do not use brakets.

'''


import re
import numpy as np
import pandas as pd

from pytmge.core import element_list
from pytmge.core.plugins import progressbar


__author__ = 'Yang LIU'
__maintainer__ = 'Yang LIU'
__email__ = 'l_young@live.cn'
__version__ = '1.0'
__date__ = '2022/3/18'


def check_format(cf: str):

    is_proper_format = True

    try:
        elements_in_cf = re.split(r'[(0-9]*[\.]?[0-9]+', cf)
        if elements_in_cf[-1] == '':
            elements_in_cf = elements_in_cf[:-1]
        else:
            is_proper_format = False
        if elements_in_cf == '':
            is_proper_format = False
        for e in elements_in_cf:
            if e not in element_list:
                is_proper_format = False
                break
    except Exception:
        is_proper_format = False

    if not is_proper_format:
        print('\nchemical formula seems not right :', cf)

    return is_proper_format


def extract_composition(df_dataset):
    '''
    Extracting composition from chemical formulas.

    Parameters
    ----------
    df_dataset : DataFrame
        dataset of chemical formula and target variable.
        chemical formulas as index.

    Returns
    -------
    df_composition : DataFrame
        composition.
        chemical formulas as index, elements as columns.

    '''

    print('\n' + 'extracting composition of chemical formulas ...')

    cfs = list(df_dataset.index)

    dict_composition = {}
    error_list = []
    for i, cf in enumerate(cfs):

        if pd.isnull(cf):
            print('chemical formula No.', i + 1, 'is null ...')
        else:
            elements_in_cf = re.split(r'[(0-9]*[\.]?[0-9]+', cf)
            contents_in_cf = list(map(float, re.findall(r'[0-9]*[\.]?[0-9]+', cf)))

            is_proper_format = check_format(cf)

            if is_proper_format:
                # if the cf is an alloy, the sum of contents is close to 100, then normalize to 1.
                if 99.5 <= np.nansum(contents_in_cf) <= 100.5:
                    contents_in_cf = list(pd.Series(contents_in_cf) / 100)

                dict_composition[cf] = {}
                for e, c in zip(elements_in_cf, contents_in_cf):
                    dict_composition[cf][e] = dict_composition[cf].get(e, 0.0) + c
                    # Note: sometimes some elements appear multiple times in a cf.
            else:
                print('\nchemical formula No.', i + 1, 'seems not right :', cf)
                error_list += [cf]

        progressbar(i + 1, len(cfs))

    print('getting DataFrame from dict ...')
    df_0 = pd.DataFrame(index=list(dict_composition.keys()), columns=element_list).fillna(0)
    df_composition = pd.DataFrame.from_dict(dict_composition, orient='index')
    df_composition = (df_0 + df_composition).replace(0, np.nan)
    df_composition = df_composition.loc[:, element_list] * 1
    dict_composition = df_composition.fillna(0).to_dict(orient='index')

    # print('saving...')
    # df_composition.to_csv(str(Path(__file__).absolute().parent) + '\\' + 'df_composition.csv')
    # print('df_composition.csv')
    # with open(_path + 'dict_composition.json', 'w') as _f:
    #     json.dump(dict_composition, _f)
    # print('dict_composition.json')

    return df_composition
