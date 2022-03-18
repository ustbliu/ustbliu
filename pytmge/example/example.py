# -*- coding: utf-8 -*-

"""
An example.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from pytmge.core import elemental_data, electron_orbital_attribute
from pytmge.core.crystal import data_preparation
from pytmge.core.crystal import extract_composition
from pytmge.core.crystal import feature_design


if __name__ == '__main__':

    _path = str(Path(__file__).absolute().parent) + '\\'

    df_example = pd.read_csv(_path + 'example.csv', index_col=0).iloc[:3, :]

    eoa = electron_orbital_attribute  # refresh data

    # composition
    df_chemical_composition = extract_composition(df_example)
    df_chemical_composition.to_csv(_path + 'df_chemical_composition.csv')

    # category and subset
    dict_category = data_preparation().categorization_by_composition(df_example)
    df_subset = data_preparation().subset(df_example)

    # distances in composition space
    df_distances_inside_dataset = data_preparation().distances_within_dataset(df_example)

    df_example_0 = pd.read_csv(_path + 'example.csv', index_col=0).iloc[:100, :]
    df_example_1 = pd.read_csv(_path + 'example.csv', index_col=0).iloc[200:400, :]
    df_distances_extrapolation = data_preparation().distances_between_datasets(df_example_0, df_example_1)

    # two self-defined parameters, can be used in accessing the generalization ability of ML models.
    extrapolation_distances = pd.Series(
        df_distances_extrapolation.shape[0] / np.nansum(1 / df_distances_extrapolation, axis=1),
        index=df_distances_extrapolation.index, name='distance'
    )
    extrapolation_distance = len(extrapolation_distances) / np.nansum(1 / extrapolation_distances)

    # ------
    dict_orbital_attributes_of_shells = elemental_data.orbital_attributes_of_shells
    df_orbital_attributes_of_elements = elemental_data.orbital_attributes_of_elements

    df_features = feature_design.get_features(df_example)
    df_features.to_csv(_path + 'df_features.csv')
    df_usable_feature = feature_design.delete_unusable_features(df_features)
    df_usable_feature.to_csv(_path + 'df_usable_feature.csv')
    # df_correlation_matrix = feature_design.get_correlation_matrix(df_usable_feature)
