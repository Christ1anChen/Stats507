# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:32:06 2021

@author: Chen Guixian
"""

import pandas as pd
import pickle

# read files, retain object variables and add "cohort" column
file1 = pd.read_sas("DEMO_G.XPT")
main1 = file1.loc[:, ["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
                    "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
main1.insert(main1.shape[1], "cohort", ["G" for i in range(main1.shape[0])])

file2 = pd.read_sas("DEMO_H.XPT")
main2 = file2.loc[:, ["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
                    "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
main2.insert(main2.shape[1], "cohort", ["H" for i in range(main2.shape[0])])

file3 = pd.read_sas("DEMO_I.XPT")
main3 = file3.loc[:, ["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
                    "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
main3.insert(main3.shape[1], "cohort", ["I" for i in range(main3.shape[0])])

file4 = pd.read_sas("DEMO_J.XPT")
main4 = file4.loc[:, ["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
                    "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]]
main4.insert(main4.shape[1], "cohort", ["J" for i in range(main4.shape[0])])

result1 = pd.concat([main1, main2, main3, main4])
# convert "ages" less than 1 to 0
for i in range(result1.shape[0]):
    age = result1.iloc[i,1]
    if age != int(age):
        result1.iloc[i,1] = int(age)
result1 = result1.convert_dtypes()

# convert missing data to -1    
result1["DMDEDUC2"] = result1["DMDEDUC2"].apply(lambda x: -1 if pd.isnull(x)
                                                else x)
result1["DMDMARTL"] = result1["DMDMARTL"].apply(lambda x: -1 if pd.isnull(x)
                                                else x)
# rename columns
result1 = result1.rename(columns={"SEQN": "unique_ids", "RIDAGEYR": "age",
                                "RIDRETH3": "race_ethnicity", "DMDEDUC2": "education",
                                "DMDMARTL": "marital_status", "RIDSTATR": "weight1", 
                                "SDMVPSU": "weight2", "SDMVSTRA": "weight3", 
                                "WTMEC2YR": "weight4", "WTINT2YR": "weight5"})
# save result1
file = open("demographic_data.pkl", "wb")
pickle.dump(result1, file)
file.close()




# read files and add "cohort" column
file5 = pd.read_sas("OHXDEN_G.XPT")
file5.insert(file5.shape[1], "cohort", ["G" for i in range(file5.shape[0])])

file6 = pd.read_sas("OHXDEN_H.XPT")
file6.insert(file6.shape[1], "cohort", ["H" for i in range(file6.shape[0])])

file7 = pd.read_sas("OHXDEN_I.XPT")
file7.insert(file7.shape[1], "cohort", ["I" for i in range(file7.shape[0])])

file8 = pd.read_sas("OHXDEN_J.XPT")
file8.insert(file8.shape[1], "cohort", ["J" for i in range(file8.shape[0])])

result2 = pd.concat([file5, file6, file7, file8])

# retain object variables
cols = result2.columns
mcols = ["SEQN", "OHDDESTS"]
for i in range(len(cols)):
    temp = cols[i]
    if temp[0:3]=="OHX" and temp[5:7]=="TC" and len(temp)==7:
        mcols.append(temp)
    elif temp[0:3]=="OHX" and temp[5:8]=="CTC" and len(temp)==8:
        mcols.append(temp)
    else:
        continue
mcols.append("cohort")       
result2 = result2.loc[:, mcols]  

result2 = result2.convert_dtypes()
# convert "coronal cavities" columns to standard format
for i in range(len(mcols)):
    temp = mcols[i]
    if  temp[0:3]=="OHX" and temp[5:8]=="CTC" and len(temp)==8:
        result2[temp] = result2[temp].apply(lambda x: x.replace("b", "").replace("'",""))   
result2 = result2.convert_dtypes()

def name_convert(name):
    """
    Rename the columns with literate variable names using all lower case.

    Parameters
    ----------
    name : str
        Original column name.

    Returns
    -------
    str
        Converted column name.

    """
    if name == "SEQN":
        return "unique_ids"
    elif name == "OHDDESTS":
        return "status_code"
    elif name[5:7] == "TC":
        return "tooth_counts_"+name[3:5]
    elif name[5:8] == "CTC":
        return "coronal_cavities_"+name[3:5]
    else:
        return "cohort"
# rename columns
result2 = result2.rename(columns=name_convert)  
# save result2 
file = open("oral_health_and_dentition_data.pkl", "wb")
pickle.dump(result2, file)
file.close()

print("Number of cases in 'demographic dataset':", result1.shape[0])
print("Number of cases in 'oral health and dentition dataset':", result2.shape[0])
