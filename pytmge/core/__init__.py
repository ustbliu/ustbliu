# coding: utf-8
# Copyright (c) pytmge Development Team.

"""
This package contains core modules and classes
for machine learning to predict materials.

"""


from .primary_data import elemental_data as ed
from .primary_data import electron_orbital_attribute as eoa


elemental_data = ed()
element_list = elemental_data.symbols
electron_orbital_attribute = eoa()
