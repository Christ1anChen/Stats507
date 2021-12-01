# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Problem Set 6: pd_topic_gxchen.py
# **Stats 507, Fall 2021**  
# *Guixian Chen*
# *gxchen@umich.edu*
# *November, 2021*

# ## Table Styles
# * `pandas.io.formats.style.Styler` helps style a DataFrame or Series (table) according to the data with HTML and CSS.
# * Using `.set_table_styles()` from `pandas.io.formats.style.Styler` to control areas of the table with specified internal CSS.
# * Funtion `.set_table_styles()` can be used to style the entire table, columns, rows or specific HTML selectors.
# * The primary argument to `.set_table_styles()` can be a list of dictionaries with **"selector"** and **"props"** keys, where **"selector"** should be a CSS selector that the style will be applied to, and **props** should be a list of tuples with **(attribute, value)**.

import pandas as pd
import numpy as np

# example 1
df1 = pd.DataFrame(np.random.randn(10, 4),
                  columns=['A', 'B', 'C', 'D'])
df1.style.set_table_styles(
    [{'selector': 'tr:hover',
      'props': [('background-color', 'yellow')]}]
)

# example 2
df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, np.nan],[19, 439, 6, 452, 226,232]],
                  index=pd.Index(['Tumour (Positive)', 'Non-Tumour (Negative)'], name='Actual Label:'),
                  columns=pd.MultiIndex.from_product([['Decision Tree', 'Regression', 'Random'],
                                                      ['Tumour', 'Non-Tumour']], names=['Model:', 'Predicted:']))
s = df.style.format('{:.0f}').hide_columns([('Random', 'Tumour'), ('Random', 'Non-Tumour')])
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #000066; color: white;'
}
s.set_table_styles([cell_hover, index_names, headers])
