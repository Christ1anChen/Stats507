# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

#
# ## Provided by:
#
#  __Ahmad Shirazi__<br>
#  __Shirazi@umich.edu__
#  
#
# ## Topic:
#  
# Using <font color=red>glob module</font> to create a data frame from multiple files, row wise:

# +
import pandas as pd
from glob import glob


# - We can read each dataframe from its own CSV file, combine them together and delet the original dataframes. <br>
# - This will need a lot of code and will be memory and time consuming.<br>
# - A better solution is to use the built in glob module.<br>
# - Lets make example dataframes in the next cell.<br>

# +
# making 3 dataframes as inputs for our example
data_age1 = pd.DataFrame({'Name':['Tom', 'Nick', 'Krish', 'Jack'],
        'Age':[17, 31, 28, 42]})

data_age2 = pd.DataFrame({'Name':['Kenn', 'Adam', 'Joe', 'Alex'],
        'Age':[20, 21, 19, 18]})

data_age3 = pd.DataFrame({'Name':['Martin', 'Jenifer', 'Roy', 'Mike'],
        'Age':[51, 30, 38, 25]})

# Saving dataframes as CSV files to be used as inputs for next example
data_age1.to_csv('data_age1.csv', index=False)
data_age2.to_csv('data_age2.csv', index=False)
data_age3.to_csv('data_age3.csv', index=False)


# - We pass a patern to the glob (including wildcard characters)
# - It will return the name of all files that match that pattern in the data subdirectory.
# - We can then use a generator expression to read each file and pass the results to the concat function.
# - It will concatenate the rows for us to a single dataframe.
# -

students_age_files = glob('data_age*.csv')
students_age_files

pd.concat((pd.read_csv(file) for file in students_age_files), ignore_index=True)


# ## Concatenate 
# *Dingyu Wang*
#
# wdingyu@umich.edu

# ## Concatenate 
# + Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# + Combine two Series.

import pandas as pd

s1 = pd.Series(['a', 'b', 'c'])
s2 = pd.Series(['d', 'e', 'f'])
pd.concat([s1, s2])

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.

df1 = pd.DataFrame([['a', 1], ['b', 2], ['c', 3]],
                   columns=['letter', 'number'])
df2 = pd.DataFrame([['d', 4], ['e', 5], ['f', 6]],
                   columns=['letter', 'number'])
pd.concat([df1, df2])

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.
# * Combine DataFrame objects with overlapping columns and return only those
# that are shared by passing inner to the join keyword argument
# (default outer).

df3 = pd.DataFrame([['a', 1, 'Mary'], ['b', 2, 'John'], ['c', 3, 'James']],
                   columns=['letter', 'number', 'name'])
pd.concat([df1, df3])
pd.concat([df1, df3], join='inner')

# ## Concatenate 
# * Concatenate pandas objects along a particular axis with optional set logic
# along the other axes.
# * Combine two Series.
# * Combine two DataFrame objects with identical columns.
# * Combine DataFrame objects with overlapping columns and return only those
# that are shared by passing in `join=inner`(default outer).
# * Combine DataFrame objects horizontally along the x axis by passing in
# `axis=1`(default 0).

df4 = pd.DataFrame([['Tom', 24], ['Jerry', 18], ['James', 22]],
                   columns=['name', 'age'])
pd.concat([df1, df4])
pd.concat([df1, df4], axis=1)

# # Sparse Data Structures - falarcon@umich.edu
#
# Felipe Alarcon Pena

# `pandas` offers a way to speed up not only calculations in the typical `sparse` meaning, i.e. , `DataFrames` with 0's, but also for particular values or `NaN's`.
#
#
# Let's first start showing the effect it has on discarding `NaN's` or a particular values and compare it with other methods. 
#
# The goal of using `sparse` Data Structures is to allocate memory efficiently in large data sets and also speed-up possible operations between `sparse` Data Structures. `Sparse Data Structure` saved values and locations instead of the whole Dataframe. 
#

import pandas as pd
import numpy as np

# +
## Creating toy data.

array_nans = np.random.randn(500, 10000)

# Switching some values to NaN's to produce a sparse structure.
array_nans[10:499,1:9900] = np.nan

dense_df = pd.DataFrame(array_nans)
sparse_df = dense_df.astype(pd.SparseDtype("float", np.nan))

print(" Density of sparse DataFrame: "+str(sparse_df.sparse.density))
# -

# ## Efficiency in storing Sparse DataStructures
#
# `Sparse DataStructures` are more efficient in allocating memory for large datasets with lots of NaN's or information that it is not of interest. The toy data has some sparsity $\sim$ 50%, but real data or matrices could have densities of the orden of $\sim$0.1 % or less.

# +
## Let's compare the storing times for different methods and the same datastructure  being sparse or not.

print('Dense data structure : {:0.3f} bytes'.format(dense_df.memory_usage().sum() / 1e3))
print('Sparse data structure : {:0.3f} bytes'.format(sparse_df.memory_usage().sum() / 1e3))
# -

# Even though the sparse allocate memory better, thy take slightly longer to be created. Nevertheless, we will prove that when there are heavy operations being undertaken in large sparse data structures, the speed-up is worth it, and the allocation of memory as well.

# %timeit  df_2 = pd.DataFrame(array_nans)

# %timeit  sparse_df = pd.DataFrame(array_nans).astype(pd.SparseDtype("float", np.nan))

# ## Speed-up of calculations in Sparse DataStructures and comparison with scipy.
#
# Finally we compare the time it takes to operate on `Dense DataStructures` and `Sparse DataStructures`. Operating directly on `Sparse DataStructures` is not really efficient because `pandas` converts them to `Dense DataStructures` first. Nevertheless the `scipy` package has methods that take advantage of the psarsity of matrices to speed-up those processes.

# +
## scipy also offers methods for sparse arrays, although in the full with 0's meaning,
from scipy.sparse import csr_matrix

rnd_array = np.zeros((10000,500))
rnd_array[200:223,13:26] = np.random.randn(23,13)
sparse_rnd_array = csr_matrix(rnd_array)
sparse_matrix_df = csr_matrix(sparse_df)
# -

# %timeit sparse_matrix_df.dot(sparse_rnd_array)

# %timeit dense_df.dot(rnd_array)

# As we can see `Sparse` methods are specially useful to manage data with repeated values or just values we are not interested in. It can also be used to operate on them using `scipy` and its methods for `sparse` arrays, which could be much faster that standard multiplications. It is important to notice that it is only faster when the sparsity is significant, usually less than 1%.


# ## Topics in Pandas
# **Stats 507, Fall 2021** 

# ## Contents
# + [Pandas Table Visualization](#Pandas-Table-Visualization)
# + [Sorting in Python Pandas](#Sorting-in-Python-Pandas)
# + [Python Classes and Objects](#Python-Classes-and-Objects)

# ## Pandas Table Visualization
# **Author:** Cheng Chun, Chien  
# **Email:**　jimchien@umich.edu  
# [PS6](https://jbhender.github.io/Stats507/F21/ps/ps6.html)

# ## Introduction
# - The slide shows visualization of tabular data using the ***Styler*** class
# - The ***Styler*** creates an HTML table and leverages CSS styling language to control parameters including colors, fonts, borders, background, etc.
# - Following contents will be introduced.
#     1. Formatting Values
#     2. Table Styles
#     3. Bulitin Styles

# ## Formatting Values
# - Styler can distinguish the ***display*** and ***actual*** value
# - To control the display value, use the *.format()* method to manipulate this according to a format spec string or a callable that takes a single value and returns a string.
# - Functions of *.format()*
#     - *precision*: formatting floats
#     - *decimal / thousands*: support other locales
#     - *na_rep*: display missing data
#     - *escape*: displaying safe-HTML or safe-LaTeX

# ## Table Styles
# - Recommend to be used for broad styling, such as entire rows or columns at a time.
# - 3 primary methods of adding custom CSS styles
#     - *.set_table_styles()*: control broader areas of the table with specified internal CSS. Cannot be exported to Excel.
#     - *.set_td_classes()*: link external CSS classes to data cells. Cannot be exported to Excel.
#     - *.apply() and .applymap()*: add direct internal CSS to specific data cells. Can export to Excel.
# - Also used to control features applying to the whole table by generic hover functionality, *:hover*.
# - List of dicts is the format to pass styles to .set_table_styles().

# ## Builtin Styles
# - *.highlight_null*: identifying missing data.
# - *.highlight_min / .highlight_max*: identifying extremeties in data.
# - *.background_gradient*: highlighting cells based or their, or other, values on a numeric scale.
# - *.text_gradient*: highlighting text based on their, or other, values on a numeric scale.
# - *.bar*: displaying mini-charts within cell backgrounds.

import numpy as np
import pandas as pd
# Example of table bar chart
np.random.seed(0)
df = pd.DataFrame(np.random.randn(5,4), columns=['A','B','C','D'])
df.style.bar(subset=['A', 'B'], color='#d65f5f')

# ---
# ---
#

# ## Sorting in Python Pandas
# **Author:** Nuona Chen  
# **Email:** nuona@umich.edu 

#  <h3>DataFrame.sort_values()</h3>
#
#
#
#  <h2>DataFrame.sort_values()</h2>
#  <h3>General Form</h3>
#  <li>DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)</li>
#  
#  <h3>What it does</h3>
#  <li>sorts the input DataFrame by values</hi>
#  
#  <h3>Input Parameters</h3>
#   <li>by: name or list of names to sort by</li>
#    <li>axis: 0 -> sort indices, 1 -> sort columns </li>
#    <li>ascending: true -> sort ascending, false -> sort descending</li>
#    <li>inplace: true -> sort in-place </li>
#    <li>kind: sorting algorithm </li>
#    <li>na_position: first -> put NaNs at the beginning, last -> put NaNs at the end</li>
#    <li>ignore_index: true -> label axis as 0, 1,..., n-1</li>
#  
#  <h3>Function Output</h3>
#  <li>The sorted Pandas DataFrame</li>
#  <h4>Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html</h4>

# +
from IPython.core.display import HTML, display
display(HTML('<h2>Example1 - Sort by Columns</h2>'))

data = {"id": [1, 2, 3, 4, 5, 6],
        "gpa": [3.2, 3.7, 3.1, 3.0, 3.7, 3.7],
        "total credit hours": [55, 68, 100, 94, 46, 110],
        "major": ["Statistics", "Anthropology", "Business", "Computer Science", "Business", "Statistics"]}
print("Input DataFrame - ie_data: ")
ie_data = pd.DataFrame(data)
print(ie_data)

display(HTML('<h3>Sort by gpa in an ascending order</h3>'))
display(HTML("<code>ie_data.sort_values(by='gpa', ascending=True)</code>"))

print(ie_data.sort_values(by="gpa", ascending=True))

display(HTML('<h3>Sort by gpa and major in an descending order</h3>'))
display(HTML("<code>ie_data.sort_values(by=['gpa', 'major'], ascending=[False, True])</code>"))

print(ie_data.sort_values(by=["gpa", "major"], ascending=[False, True]))
display(HTML('<p>The order of variables in the by statement is the order of sorting. Major is sorted after gpa is sorted.</p>'))

display(HTML('<h2>Example2 - Sort by Rows</h2>'))

del ie_data["major"]
print("Input DataFrame - ie_data: ")
print(ie_data)
display(HTML('<h3>Sort by the row with index = 4 in an descending order</h3>'))
display(HTML("<code>ie_data.sort_values(by=4, axis=1, ascending=False)</code>"))

ie_data.sort_values(by=4, axis=1, ascending=False)


# -

# ---
# ---
#

# ## Python Classes and Objects
#
# **Name:** Benjamin Osafo Agyare  
# **UM email:** bagyare@umich.edu  
#

# ## Overview
#   - [Python classes](#Python-classes)
#   - [Creating a Class data](#Creating-a-Class)
#   - [The __init__() Function](#The-__init__()-Function)
#   - [Object Methods and the self operator](#Object-Methods-and-the-self-operator)
#   - [Modifying and Deleting objects](#Modifying-and-Deleting-objects)

# ## Python classes
# - Python is an object oriented programming language.
# - Almost everything in Python is an object, with its properties and methods.
# - A Class is like an object constructor, or a "blueprint" for creating objects.

# ## Python classes
# - Some points on Python class:
#   + Classes are created by keyword class.
#   + Attributes are the variables that belong to a class.
#   + Attributes are always public and can be accessed using the dot (.) operator. Eg.: Myclass.Myattribute

# ## Creating a Class
# - To create a class in python, use the reserved keyword `class`
#
# ### Example
# - Here is an example of creating a class named Student, with properties name, school and year:

class Student:
    school = "U of Michigan"
    year = "freshman"


# ## The __init__() Function
#
# - To make better meaning of classes we need the built-in `__init__()` function.
# - The `__init__()` is used function to assign values to object properties, or other operations that are necessary to do when the object is being created.
# - All classes have this, which is always executed when the class is being initiated.

# +
class Student:
    def __init__(self, name, school, year):
        self.name = name
        self.school = school
        self.year = year

person1 = Student("Ben", "University of Michigan", "freshman")
print(person1.name)
print(person1.school)
print(person1.year)


# -

# ## Object Methods and the self operator
# - Objects in python can also have methods which are functions that belong to the object

# ## Object Methods and the self operator
# - The self parameter
#   + The self parameter `self` in the example above references the current state of the class.
#   + It can be used to access the variables that belong to the class.
#   + You can call it whatever you like but has to be the first parameter of any function in a class

# ## Example

# +
class Student:
    def __init__(myself, name, school, year):
        myself.name = name
        myself.school = school
        myself.year = year

    def getAttr(person):
        print("Hi, I'm " + person.name + " and a "
              + person.year + " at " + person.school)


person2 = Student("Ben", "University of Michigan", "freshman")
person2.getAttr()
# -

# ## Modifying and Deleting objects
# - You can modify object properties
# - It is also possible to delete properties objects using the `del` keyword and same can be used to delete the object itself

# ## Example

# +
## Modify object properties
person2.year = "senior"

## Delete properties of objects
del person2.name

## Delete object
del person2
# -

# ## References:
# - [https://www.w3schools.com/python](https://www.w3schools.com/python)
# - [https://www.geeksforgeeks.org/python-classes-and-objects/](https://www.geeksforgeeks.org/python-classes-and-objects/)
# - [https://www.tutorialspoint.com/python3](https://www.tutorialspoint.com/python3)

# ---
# ---


import pandas as pd
import numpy as np
import random

# **Stan Brouwers** *brouwers@umich.edu*

# # Pandas, write to LaTeX
#
# - Pandas function.
# - Can write dataframe to LaTeX table.
# - Helpful when writing reports.
# - Easy to call.
#

# # A first example
#
# - Multiple ways to call the function.
# - Selecting certain parts of a dataframe also possible.
# - Column names and indices are used to label rows and columns.

# Generate some data.
data = pd.DataFrame([[random.randint(1,10) for columns in range(5)] for rows in range(5)])
print(data.to_latex())

# # Another example
#
# - Table can be written to a file. (buf input)
# - Subset of columns can be chosen.
#     - Using list of column names.

data.to_latex(buf='./test.doc')
# Only printing certain columns
print(data.to_latex(columns=[0, 4]))


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# + [Duplicate labels](#Duplicate-labels) 
# + [Topic 2 Title](#Topic-2-Title)
#
# # Duplicate labels
#
# **Xinyi Liu**
# **xyiliu@umich.edu**

import numpy as np
import pandas as pd

# * Some methods cannot be applied on the data series which have duplicate labels (such as `.reindex()`, it will cause error!),
# * Error message of using the function above: "cannot reindex from a duplicate axis".

series1 = pd.Series([0, 0, 0], index=["A", "B", "B"])
#series1.reindex(["A", "B", "C"]) 

# * When we slice the unique label, it returns a series,
# * when we slice the duplicate label, it will return a dataframe.

df1 = pd.DataFrame([[1,1,2,3],[1,1,2,3]], columns=["A","A","B","C"])
df1["B"]
df1["A"]

# * Check if the label of the row is unique by apply `index.is_unique ` to the dataframe, will return a boolean, either True or False.
# * Check if the column label is unique by `columns.is_unique`, will return a boolean, either True or False.

df1.index.is_unique
df1.columns.is_unique


# * When we moving forward of the data which have duplicated lables, to keep the data clean, we do not want to keep duplicate labels. 
# * In pandas version 1.2.0 (cannot work in older version), we can make it disallowing duplicate labels as we continue to construct dataframe by `.set_flags(allows_duplicate_labels=False)`
# * This function applies to both row and column labels for a DataFrame.

# +
#df1.set_flags(allows_duplicate_labels=False) 
## the method above cannot work on my end since my panda version is 1.0.5
# -

# Reference: https://pandas.pydata.org/docs/user_guide/duplicates.html

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
#
# + [Function](Function) 
# + [Topic 2 Title](#Topic-2-Title)
#
# ## Function
# Defining parts of a Function
#
# **Sachie Kakehi**
#
# sachkak@umich.edu

# ## What is a Function
#
#   - A *function* is a block of code which only runs when it is called.
#   - You can pass data, known as *parameters* or *arguments*, into a function.
#   - A function can return data as a result.

# ## Parts of a Function
#
#   - A function is defined using the $def$ keyword
#   - Parameters or arguments are specified after the function name, inside the parentheses.
#   - Within the function, the block of code is defined, often $print$ or $return$ values.

# ## Example
#
#   - The following function multiplies the parameter, x, by 5:
#   - Note : it is good practice to add a docstring explaining what the function does, and what the parameters and returns are. 

def my_function(x):
    """
    The function multiplies the parameter by 5.
    
    Parameters
    ----------
    x : A float or integer.
    
    Returns
    -------
    A float or integer multiplied by 5. 
    """
    return 5 * x
print(my_function(3))

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
# + [Duplicate labels](#Duplicate-labels) 
# + [Topic 2 Title](#Topic-2-Title)
# # Duplicate labels
#
# **Shuyan Li**
# **lishuyan@umich.edu**
#
# Real-world data is always messy. Since index objects are not required to be unique, sometimes we can have duplicate rows or column labels. 
# In this section, we first show how duplicate labels change the behavior of certain operations. Then we will use pandas to detect them if there are any duplicate labels, or to deal with duplicate labels.
#
# - Consequences of duplicate labels
# - Duplicate label detection
# - Deal with duplicate labels

import pandas as pd
import numpy as np

# Generate series with duplicate labels
s1 = pd.Series([0,4,6], index=["A", "B", "B"])

# ## Consequences of duplicate labels
# Some pandas methods (`Series.reindex()` for example) don’t work with duplicate indexes. The output can’t be determined, and so pandas raises.

s1.reindex(["A", "B", "C"])

# Other methods, like indexing, can cause unusual results. Normally indexing with a scalar will reduce dimensionality. Slicing a DataFrame with a scalar will return a Series. Slicing a Series with a scalar will return a scalar. However, with duplicate labels, this isn’t the case.

df1 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=["A", "A", "B"])
df1

# If we slice 'B', we get back a Series.

df1["B"] # This is a series

# But slicing 'A' returns a DataFrame. Since there are two "A" columns.

df1["A"] # This is a dataframe

# This applies to row labels as well.

df2 = pd.DataFrame({"A": [0, 1, 2]}, index=["a", "a", "b"])
df2

df2.loc["b", "A"]  # This is a scalar.

df2.loc["a", "A"]  # This is a Series.

# ## Duplicate Label Detection
# We can check whether an Index (storing the row or column labels) is unique with `Index.is_unique`:

df2.index.is_unique # There are duplicate indexes in df2.

df2.columns.is_unique # Column names of df2 are unique.

# `Index.duplicated()` will return a boolean ndarray indicating whether a label is repeated.

df2.index.duplicated()

# ## Deal with duplicate labels
# - `Index.duplicated()` can be used as a boolean filter to drop duplicate rows.

df2.loc[~df2.index.duplicated(), :]

# - We can use `groupby()` to handle duplicate labels, rather than just dropping the repeats. 
#
# For example, we’ll resolve duplicates by taking the average of all rows with the same label.

df2.groupby(level=0).mean()

# Reference: https://pandas.pydata.org/docs/user_guide/duplicates.html

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   
#
# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Empty cells](#Empty-cells)
# + [Windows Rolling](#Windows-Rolling) 
# + [Data Transformation](#Data-Transformation)


# ## Empty cells
# ---
# **Name: Yan Xu**
#
# Email:yanyanxu@umich.edu

import pandas as pd
import numpy as np
from datetime import datetime
from ps1_solution import ci_prop
import numpy.random as npra
from warnings import warn
import matplotlib.pyplot as plt
from os.path import exists


from scipy import stats
from scipy.stats import chi2_contingency 
from IPython.display import HTML

# Remove rows: remove rows that contain empty cells. Since data sets can be very big, and removing a few rows will not have a big impact on the result.
# Replace Empty Values:insert a new value using fillna() to replace NA.
# Replace Only For a Specified Columns: To only replace empty values for one column, specify the column name for the DataFrame.

#If you want to consider inf and -inf to be “NA” in computations
pd.options.mode.use_inf_as_na = True

df = pd.read_csv('https://www.w3schools.com/python/pandas/dirtydata.csv.txt')
df #The dataframe containing bad data we want to clean

# To make detecting missing values easier (and across different array dtypes), pandas provides the isna() and notna() functions, which are also methods on Series and DataFrame objects

df["Date"][20:25].notna()

new_df = df.dropna()
print(new_df.to_string())#dropna() method returns a new DataFrame, and will not change the original.
#If you want to change the original DataFrame, use the `inplace = True` argument

#insert a new value to replace the empty values
df.fillna(130)
df["Calories"].fillna(130, inplace = True)#only replace empty values for one column


# ### Data in wrong format
#
# In our Data Frame, we have two cells with the wrong format.
# Check out row 22 and 26, the 'Date' column should be a string that represents a date,try to convert all cells in the 'Date' column into dates.

# Method to validate a date string format in Python


date_string = '12-25-2018'
format = "%Y/%m/d"

try:
  datetime.strptime(date_string, format)
  print("This is the correct date string format.")
except ValueError:
  print("This is the incorrect date string format. It should be YYYY/MM/DD")


# This is the incorrect date string format. It should be YYYY/MM/DD

# for row 26,the "date" column is in wrong format
df['Date'] = pd.to_datetime(df['Date'])


# ### Removing Duplicates
#
# Duplicate rows are rows that have been registered more than one time.
# To discover duplicates, we can use the duplicated() method.
# The duplicated() method returns a Boolean values for each row.

print(df[10:15].duplicated())


# To remove duplicates, use the drop_duplicates() method.

df.drop_duplicates(inplace = True)  


# ## Windows Rolling
# ---
# **Name: Junyuan Yang**
#
# **UM email: junyyang@umich.edu**
#
# Return a rolling object allowing summary functions to be applied to windows of length n.
# By default, the result is set to the right edge of the window. This can be changed to the center of the window by setting center=True.
# Each points' weights could be determined by win_type shown in windows function, or evenly weighted as default.

import numpy as np
import pandas as pd
from os.path import exists
import re

rng = np.random.default_rng(9 * 2021 * 28)
n=100
a = rng.binomial(n=1, p=0.5, size=n)
b = 1 - 0.5 * a + rng.normal(size=n)
c = 0.8 * a + rng.normal(size=n) 
df = pd.DataFrame({'a': a, 'b': b, 'c': c})
df['c'].plot()

# - Calculating the mean in centered windows with a window length of 10 and windows type of 'triangular'

df['c'].rolling(10, center=True, win_type='triang').mean().plot()

# - Except existing functions like `sum`, `mean` and `std`, you could also use the self defined funciton by `agg.()`

df['c'].rolling(10).agg(lambda x: max(x)).plot()


# ## Data Transformation 
# ---
# **Name: Tong Wu**
#
# **UM email: wutongg@umich.edu**  
#
# Data transforming technique is important and useful when we prepare the data
# for analysis.  
# - Removing duplicates
# - Replacing values
# - Discretization and binning
# - Detecting and filtering outliers
# - Computing indicator/dummy variables
# ### Removing duplicates
# - Use `duplicated()` method returning a boolean Series to indicate which row
# is duplicated or not.
# - Duplicated rows will be dropped using `drop_duplicates()` when the
# duplicated arrary is `False`.

import pandas as pd
import numpy as np
import scipy.stats as st

data = pd.DataFrame({
    'a': [('red', 'black')[i % 2] for i in range(7)],
    'b': [('x', 'y', 'z')[i % 3] for i in range(7)]
    })
data.duplicated()
data.drop_duplicates()

# - We can specify a subset of data to detect duplicates.

data.drop_duplicates(['a'])

# - Above methods by default keep the first observed duplicated, but we can 
# keep the last occurance and drop the first occurance.

data.drop_duplicates(['a', 'b'], keep='last')

# ### Replacing values
# - General replacing approach
#  + When we find flag values for missing value, we can replace them with NAs.

pd.Series([1., -999., 2., -999., 5., 3.]).replace(-999, np.nan)

# - In a special case, we need to detect missing values and fill in them. 
#  +  Built-in Python `None` value is also treated as `NA`.

data1 = pd.DataFrame(np.random.randn(6,3))
data1.iloc[:3, 1] = np.nan
data1.iloc[:2, 2] = np.nan
data1.iloc[1, 0] = None

# Detect missing values by rows and drop rows with all NAs.
data1.dropna(axis=0, how='all')
# For time seris data, we want to keep rows with obervations.
data1.dropna(thresh=2)

# Fill in missing values.
# - Note that `fillna()` method return a new object by default.
#  + Using `inplace=True` to modify the existing object.
# - `ffill` method propagate last valid observation forward.

data1.fillna(0)
_ = data1.fillna(method='ffill', inplace=True)

# ### Discretization and binning
# This technique is used when we want to analyze continuous data seperated
# into different bins.  
# For example, we have a group of people and the **age** isgrouped into bins.

ages = [20, 17, 25, 27, 21, 23, 37, 31, 61, 45, 41, 88]
# Default it is cut into intervals with left side opened and right side closed
bins = [15, 25, 35, 60, 90]
cats = pd.cut(ages, bins)
# Categorical object
cats
cats.codes
cats.categories
# Bins count
pd.value_counts(cats)

# - Cut without emplicit bin edges.
#  + It will compute equal-length bins using the range of data.

pd.cut(ages, 4, precision=2)

# - Cut data based on sample quantiles
cat2 = pd.qcut(np.random.randn(1000), 4, precision=2)
pd.value_counts(cat2)


# ### Detecting and filtering outliers
# Here is an example with normal distributed data.
data2 = pd.DataFrame(np.random.randn(1000, 4))
data2.describe()

# Find rows which contains absolute value is larger than 3.
data2[(np.abs(data2) > 3).any(1)]
# Cap values outside the interval -3 to 3
data2[(np.abs(data2) > 3)] = np.sign(data2) * 3
data2.describe()

# ### Computing indicator/dummy variables
# We can convert a categorical variable into an indicator matrix. That is if
# a column contains $k$ distinct values, the indicator matrix is derived with
# $k$ colunms with 1s and 0s.

pd.get_dummies(cats)

# ## Problem Set 6: pd_topic_nkernik.py
# **Stats 507, Fall 2021**  
# *Nathaniel Kernik*
# nkernik@umich.edu
# *November, 2021*

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd

# 79: -------------------------------------------------------------------------

# ## Question 0 - Topics in Pandas
# Data in/out - Reading multiple files to create a single DataFrame


for i in range(3):
    data = pd.DataFrame(np.random.randn(10, 4))
    data.to_csv("file_{}.csv".format(i))

files = ["file_0.csv", "file_1.csv", "file_2.csv"]
result = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(result)


# Basic grouping with apply

df = pd.DataFrame(
    {
        "animal": "cat dog cat fish dog cat cat".split(),
        "size": list("SSMMMLL"),
        "weight": [8, 10, 11, 1, 20, 12, 12],
        "adult": [False] * 5 + [True] * 2,
    }
)
print(df)

# List the size of the animals with the highest weight.

df.groupby("animal").apply(lambda subf: subf["size"][subf["weight"].idxmax()])

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
print(df)

# Using both row labels and value conditionals

df[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))]

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]},
    index=["foo", "bar", "boo", "kar"],
)


# Slicing methods
#     positional-oriented
#     label-oriented

print(df.loc["bar":"kar"])
print(df[0:3])
print(df["bar":"kar"])


# Creating new columns using applymap
df = pd.DataFrame({"AAA": [1, 2, 1, 3], "BBB": [1, 1, 2, 2], "CCC": [2, 1, 3, 1]})
print(df)

source_cols = df.columns
new_cols = [str(x) + "_cat" for x in source_cols]
categories = {1: "Alpha", 2: "Beta", 3: "Charlie"}
df[new_cols] = df[source_cols].applymap(categories.get)
print(df)


# ## Problem Set 6: pd_topic_gxchen.py
# **Stats 507, Fall 2021**  
# *Guixian Chen*
# gxchen@umich.edu
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
