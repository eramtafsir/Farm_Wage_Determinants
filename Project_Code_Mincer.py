# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:54:02 2023

@author: eramt
"""


from IPython import get_ipython;   
get_ipython().magic('reset -sf')


import pandas as pd
import os
import numpy as np
import wooldridge as woo
import statsmodels.formula.api as smf
import math
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import compare



os.chdir('C:/Users/eramt/OneDrive/Desktop/Ecotrix/Ecotrix project/My project/Data sets/Python Code Housing Project/Project submission/Methods Directory')



df1 = pd.read_csv('NAWS_A2E191.csv')
df2 = pd.read_csv('NAWS_F2Y191.csv')

naws = pd.concat([df1, df2], axis = 1, join='inner')

"""
for col in df1.columns:
    print(col)
    
for col in df2.columns:
    print(col)

print('\n\n')
"""

nawsAE = df1.filter(items=['D05','D06','D08','A09','B11','FY','CROP'])
nawsFY = df2.filter(items=['REGION6','Worktype'])



a = nawsAE.dtypes
print(f'{a}')

b = nawsFY.dtypes
print(f'{b}')


naws = pd.concat([nawsAE, nawsFY], axis = 1)

c = naws.dtypes
print(f'{c}')





# Selecting our Data
#########################################################################################

# Mincer Earnings Regression
# Wage income as a function of Schooling and Experience


# Schooling and Experience
# A09 - What is the Highest Grade in School You/They completed? (If completed "GED" enter 12)
# B11 - Approximately how many years have you done FARM WORK in the U.S.A.? [COUNT  ANY YEAR IN WHICH 15 DAYS OR MORE WERE WORKED.] 


# Wage
# D5 - Can you tell me the amount that your current employer paid you on your last day (cash or check) After taxes
# D6 - Can you tell me the amount that your current employer paid you on your last day (cash or check) Before taxes:


# D8 - How many hours did you work during that period? [ONE DAY/ ONE WEEK/ TWO WEEKS/ ONE MONTH/ OTHER.) Hours

# Other Factors
# REGION6 - Which region did the interview take place in. We will talk more about this later and likely in our presentation. (5 is Northwest)
# Worktype - Type of Work (1 Field Work, 2 Nursery, 3 Packing House, 7 Other)


# We need to make a variable for CPI, the first step to that is finding out what year the data was taken
# FY - Fiscal Year (Data goes back to 1989)
cpi = pd.read_csv('cpi.csv')

########################################################################################



# Cleaning the Data
########################################################################################



#CPI
cpi = cpi.rename(columns={'year' : 'FY'})

naws = naws.merge(cpi, how = 'left')

c = naws.dtypes
print(f'{c}')

naws = naws.rename(columns={'cpi v' : 'cpi'})

#CPI Multiplier
naws['cpimult'] = naws['cpi'].iloc[-1] / naws['cpi']


#naws = naws.apply(pd.to_numeric, errors='coerce')


# Removing Messy Data ($9999) in D06
# Replacing with NAN

naws['D06'] = pd.to_numeric(naws['D06'], errors = 'coerce')

# naws = naws.replace(9999, np.NaN)

c = naws.dtypes
print(f'{c}')

naws['D06'] = naws['D06'].astype(str)

c = naws.dtypes
print(f'{c}')



# Choosing the Range of Years for our Analysis
########################################################################################

c = naws.dtypes
print(f'{c}')

naws['FY'] = naws['FY'].astype(np.float64)

# Data from 1989 to 2018

start_date = 2009

# end_date = '2018'


mask = (naws['FY'] >= start_date)
naws = naws.loc[mask]
naws = naws.reset_index()


# Converting objects to floats
########################################################################################

naws['D06'] = naws['D06'].astype(np.float64)

naws['D05'] = naws['D05'].str.replace(',','')
naws['D05'] = naws['D05'].astype(np.float64)

naws['REGION6'] = naws['REGION6'].astype(np.float64)

naws['index'] = naws['index'].astype(np.float64)


c = naws.dtypes
print(f'{c}')





# Wage and Adjusting Wage for Inflation
#######################################################################################

# After Taxes Wage
# D5 / D8


naws['wat']= naws['D05']/naws['D08']



# Adjusting Wage for Inflation (cpi)

naws['wati'] = naws['wat']*naws['cpimult']

########################################################################################



# First we will do a regression of all types of workers with the entirety of the United States.
########################################################################################

# Regression Model:
    # ln(Wage) = B0 + B1 * (Schooling) + B2 * (Experience) + B3 * (Experience)^2
    # ln(wati) = B0 + B1 * (A09) + B2 * (B11) + B3 * (B11)^2
    

naws = naws.rename(columns={'D05':'amntpaidat','D06':'amntpaidbt','D08':'hoursworked','A09':'education','B11':'yearsfarmwork','B11sq':'yearsfarmworksq'})

naws['yearsfarmworksq'] = naws['yearsfarmwork'] ** 2
naws['lwati'] = np.log(naws['wati'])

nawssummary = naws.describe()
print(f'\nNaws Summary:\n{nawssummary}')


x = naws[['education','yearsfarmwork','yearsfarmworksq']]
y = naws['lwati']

# with statsmodels
x = sm.add_constant(x) # adding a constant


 
model = sm.OLS(y, x, missing ='drop').fit()
predictions = model.predict(x)

 
print_model = model.summary()



##########################################################################

#Age - Age of person surveyed
#English Speaking Proficiency - B07 - From 1 to 10
#Legal Status - 1 Unauthorized, 2 Otherwork Authorized, 3 Work Authorized, 4 Legal
#EMPLOYED - Type of Employer - 2 Contract directly with Grower, 1 Farm Labor Contracted

variables = df1.filter(items=['AGE','B07','currstat','EMPLOYED'])

# Adding these Variables
naws = pd.concat([naws, variables], axis = 1)


naws = naws.rename(columns={'AGE':'age','B07':'english','EMPLOYED':'employed'})


# Creating Dummy for Current Legal Status
naws['authorized'] = np.where(naws['currstat'] < 4, 1, 0)

# Creating Dummy for whether or not the employee is contracted
naws['contract'] = np.where(naws['employed'] == 2, 1, 0)


print(naws.dtypes)
naws['authorized'] = naws['authorized'].astype(np.float64)
naws['contract'] = naws['contract'].astype(np.float64)

#################################################################################

# Crop Type Variable

naws = naws.drop(naws[naws['CROP'] == 5].index, inplace = False)
naws['fieldcrops'] = np.where(naws['CROP'] == 1, 1, 0)
naws['fruitsnuts'] = np.where(naws['CROP'] == 2, 1, 0) # BASE
naws['horticulture'] = np.where(naws['CROP'] == 3, 1, 0) 
naws['vegetables'] = np.where(naws['CROP'] == 4, 1, 0)



naws = naws.drop(naws[naws['Worktype'] == 7].index, inplace = False)
naws['fieldwork'] = np.where(naws['Worktype'] == 1, 1, 0)
naws['nursery'] = np.where(naws['Worktype'] == 2, 1, 0)
naws['packinghouse'] = np.where(naws['Worktype'] == 3, 1, 0) 



naws['east'] = np.where(naws['REGION6'] == 1, 1, 0)
naws['southeast'] = np.where(naws['REGION6'] == 2, 1, 0)
naws['midwest'] = np.where(naws['REGION6'] == 3, 1, 0)
naws['southwest'] = np.where(naws['REGION6'] == 4, 1, 0)
naws['northwest'] = np.where(naws['REGION6'] == 5, 1, 0)
naws['california'] = np.where(naws['REGION6'] == 6, 1, 0)





######################################################################33

from linearmodels import PanelOLS
import linearmodels as plm


# MODEL 1

naws = naws.set_index(['index', 'FY'], drop=False)

print('\nModel One Pooled OLS Results:\n')
reg_M1 = plm.PooledOLS.from_formula(
    formula = 'lwati ~ education + yearsfarmwork + yearsfarmworksq + C(FY)',
    data = naws)
results_M1 = reg_M1.fit()
print(f'results_M1.summary: \n{results_M1.summary}\n')



# MODEL 2


print('\nModel One Pooled OLS Results:\n')
reg_M2 = plm.PooledOLS.from_formula(
    formula = 'lwati ~ education + yearsfarmwork + yearsfarmworksq + nursery + packinghouse + fieldcrops + horticulture + vegetables + C(FY)',
    data = naws)
results_M2 = reg_M2.fit()
print(f'results_M1.summary: \n{results_M2.summary}\n')




# MODEL 3

print('\nModel One Pooled OLS Results:\n')
reg_M3 = plm.PooledOLS.from_formula(
    formula = 'lwati ~ education + yearsfarmwork + yearsfarmworksq + nursery + packinghouse + fieldcrops + horticulture + vegetables + contract + east + southeast + midwest + southwest + northwest + authorized + contract + C(FY)',
    data = naws)
results_M3 = reg_M3.fit()
print(f'results_M3.summary: \n{results_M3.summary}\n')


print('\nModel One Pooled OLS Results:\n')
reg_M3 = plm.PooledOLS.from_formula(
    formula = 'lwati ~ education + yearsfarmwork + yearsfarmworksq + nursery + packinghouse + fieldcrops + horticulture + vegetables +  east + southeast + midwest + southwest + northwest + C(FY)',
    data = naws)
results_M3 = reg_M3.fit()
print(f'results_M3.summary: \n{results_M3.summary}\n')



############################################################################
table_M1 = pd.DataFrame({'b': round(results_M1.params, 4),
                          'se': round(results_M1.std_errors, 4),
                          't': round(results_M1.tstats, 4),
                          'pval': round(results_M1.pvalues, 4)})
print(f'table_ols: \n{table_M1}\n')

table_M2 = pd.DataFrame({'b': round(results_M2.params, 4),
                         'se': round(results_M2.std_errors, 4),
                         't': round(results_M2.tstats, 4),
                         'pval': round(results_M2.pvalues, 4)})
print(f'table_re: \n{table_M2}\n')

table_M3 = pd.DataFrame({'b': round(results_M3.params, 4),
                         'se': round(results_M3.std_errors, 4),
                         't': round(results_M3.tstats, 4),
                         'pval': round(results_M3.pvalues, 4)})
print(f'table_fe: \n{table_M3}\n')

print(compare({"Model 1": results_M1, "Model 2": results_M2, "Model 3": results_M3}))