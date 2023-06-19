# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:45:27 2022

@author: hrohw
"""


import statsmodels.api as sm
import matplotlib.pyplot as plt
import patsy as pt


##############################################
##                                          ##
##               Mincer V3                  ##
##                                          ##
##############################################


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



# Job Type Bar Graph
naws['fieldwork'] = np.where(naws['Worktype'] == 1, 1, 0)
naws['nursery'] = np.where(naws['Worktype'] == 2, 1, 0)
naws['packinghouse'] = np.where(naws['Worktype'] == 3, 1, 0) 
naws['other'] = np.where(naws['Worktype'] == 7, 1, 0)


# Crop Type Bar Graph
naws['fieldcrops'] = np.where(naws['CROP'] == 1, 1, 0)
naws['fruitsnuts'] = np.where(naws['CROP'] == 2, 1, 0)
naws['horticulture'] = np.where(naws['CROP'] == 3, 1, 0) 
naws['vegetables'] = np.where(naws['CROP'] == 4, 1, 0)
naws['othercrop'] = np.where(naws['CROP'] == 5, 1, 0)


# Region Graph
naws['east'] = np.where(naws['REGION6'] == 1, 1, 0)
naws['southeast'] = np.where(naws['REGION6'] == 2, 1, 0)
naws['midwest'] = np.where(naws['REGION6'] == 3, 1, 0)
naws['southwest'] = np.where(naws['REGION6'] == 4, 1, 0)
naws['northwest'] = np.where(naws['REGION6'] == 5, 1, 0)
naws['california'] = np.where(naws['REGION6'] == 6, 1, 0)



naws2009 = naws.loc[(naws['FY'] == 2009)]
naws2010 = naws.loc[(naws['FY'] == 2010)]
naws2011 = naws.loc[(naws['FY'] == 2011)]
naws2012 = naws.loc[(naws['FY'] == 2012)]
naws2013 = naws.loc[(naws['FY'] == 2013)]
naws2014 = naws.loc[(naws['FY'] == 2014)]
naws2015 = naws.loc[(naws['FY'] == 2015)]
naws2016 = naws.loc[(naws['FY'] == 2016)]
naws2017 = naws.loc[(naws['FY'] == 2017)]
naws2018 = naws.loc[(naws['FY'] == 2018)]


total2009 = [naws2009['fieldwork'].sum(), naws2009['nursery'].sum(), naws2009['packinghouse'].sum(), naws2009['other'].sum()]
total2010 = [naws2010['fieldwork'].sum(), naws2010['nursery'].sum(), naws2010['packinghouse'].sum(), naws2010['other'].sum()]
total2011 = [naws2011['fieldwork'].sum(), naws2011['nursery'].sum(), naws2011['packinghouse'].sum(), naws2011['other'].sum()]
total2012 = [naws2012['fieldwork'].sum(), naws2012['nursery'].sum(), naws2012['packinghouse'].sum(), naws2012['other'].sum()]
total2013 = [naws2013['fieldwork'].sum(), naws2013['nursery'].sum(), naws2013['packinghouse'].sum(), naws2013['other'].sum()]
total2014 = [naws2014['fieldwork'].sum(), naws2014['nursery'].sum(), naws2014['packinghouse'].sum(), naws2014['other'].sum()]
total2015 = [naws2015['fieldwork'].sum(), naws2015['nursery'].sum(), naws2015['packinghouse'].sum(), naws2015['other'].sum()]
total2016 = [naws2016['fieldwork'].sum(), naws2016['nursery'].sum(), naws2016['packinghouse'].sum(), naws2016['other'].sum()]
total2017 = [naws2017['fieldwork'].sum(), naws2017['nursery'].sum(), naws2017['packinghouse'].sum(), naws2017['other'].sum()]
total2018 = [naws2018['fieldwork'].sum(), naws2018['nursery'].sum(), naws2018['packinghouse'].sum(), naws2018['other'].sum()]


fieldtotal = [naws2009['fieldwork'].sum(), naws2010['fieldwork'].sum(), naws2011['fieldwork'].sum(), naws2012['fieldwork'].sum(), naws2013['fieldwork'].sum(),
              naws2014['fieldwork'].sum(), naws2015['fieldwork'].sum(), naws2016['fieldwork'].sum(), naws2017['fieldwork'].sum(), naws2018['fieldwork'].sum()]
nurserytotal = [naws2009['nursery'].sum(), naws2010['nursery'].sum(), naws2011['nursery'].sum(), naws2012['nursery'].sum(), naws2013['nursery'].sum(),
              naws2014['nursery'].sum(), naws2015['nursery'].sum(), naws2016['nursery'].sum(), naws2017['nursery'].sum(), naws2018['nursery'].sum()]
packingtotal = [naws2009['packinghouse'].sum(), naws2010['packinghouse'].sum(), naws2011['packinghouse'].sum(), naws2012['packinghouse'].sum(), naws2013['packinghouse'].sum(),
              naws2014['packinghouse'].sum(), naws2015['packinghouse'].sum(), naws2016['packinghouse'].sum(), naws2017['packinghouse'].sum(), naws2018['packinghouse'].sum()]
othertotal = [naws2009['other'].sum(), naws2010['other'].sum(), naws2011['other'].sum(), naws2012['other'].sum(), naws2013['other'].sum(),
              naws2014['other'].sum(), naws2015['other'].sum(), naws2016['other'].sum(), naws2017['other'].sum(), naws2018['other'].sum()]


barWidth = 0.25
br1 = np.arange(len(fieldtotal))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]



plt.bar(br1, fieldtotal, color = 'green', width = barWidth,
        edgecolor = 'grey', label = 'Field Workers')
plt.bar(br2, nurserytotal, color = 'rebeccapurple', width = barWidth,
         edgecolor = 'grey', label = 'Nursery Workers')
plt.bar(br3, packingtotal, color = 'orange', width = barWidth,
         edgecolor = 'grey', label = 'Packinghouse Workers')
plt.bar(br4, othertotal, color = 'blue', width = barWidth,
         edgecolor = 'grey', label = 'Other Workers')

plt.xlabel('Worker Type', fontweight ='bold', fontsize = 15)
plt.ylabel('Obs. Each Year', fontweight ='bold', fontsize = 15)

plt.xticks([r + barWidth for r in range(len(fieldtotal))],
        ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'])

plt.legend()
plt.show()



######################################################################


fieldcroptotal = [naws2009['fieldcrops'].sum(), naws2010['fieldcrops'].sum(), naws2011['fieldcrops'].sum(), naws2012['fieldcrops'].sum(), naws2013['fieldcrops'].sum(),
              naws2014['fieldcrops'].sum(), naws2015['fieldcrops'].sum(), naws2016['fieldcrops'].sum(), naws2017['fieldcrops'].sum(), naws2018['fieldcrops'].sum()]
fruitsnutstotal = [naws2009['fruitsnuts'].sum(), naws2010['fruitsnuts'].sum(), naws2011['fruitsnuts'].sum(), naws2012['fruitsnuts'].sum(), naws2013['fruitsnuts'].sum(),
              naws2014['fruitsnuts'].sum(), naws2015['fruitsnuts'].sum(), naws2016['fruitsnuts'].sum(), naws2017['fruitsnuts'].sum(), naws2018['fruitsnuts'].sum()]
horticulturetotal = [naws2009['horticulture'].sum(), naws2010['horticulture'].sum(), naws2011['horticulture'].sum(), naws2012['horticulture'].sum(), naws2013['horticulture'].sum(),
              naws2014['horticulture'].sum(), naws2015['horticulture'].sum(), naws2016['horticulture'].sum(), naws2017['horticulture'].sum(), naws2018['horticulture'].sum()]
vegetablestotal = [naws2009['vegetables'].sum(), naws2010['vegetables'].sum(), naws2011['vegetables'].sum(), naws2012['vegetables'].sum(), naws2013['vegetables'].sum(),
              naws2014['vegetables'].sum(), naws2015['vegetables'].sum(), naws2016['vegetables'].sum(), naws2017['vegetables'].sum(), naws2018['vegetables'].sum()]
othercroptotal = [naws2009['othercrop'].sum(), naws2010['othercrop'].sum(), naws2011['othercrop'].sum(), naws2012['othercrop'].sum(), naws2013['othercrop'].sum(),
              naws2014['othercrop'].sum(), naws2015['othercrop'].sum(), naws2016['othercrop'].sum(), naws2017['othercrop'].sum(), naws2018['othercrop'].sum()]


barWidth = 0.20
br1 = np.arange(len(fieldcroptotal))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]



plt.bar(br1, fieldcroptotal, color = 'orange', width = barWidth,
        edgecolor = 'grey', label = 'Field Crop Workers')
plt.bar(br2, fruitsnutstotal, color = 'rebeccapurple', width = barWidth,
         edgecolor = 'grey', label = 'Fruits and Nuts Workers')
plt.bar(br3, horticulturetotal, color = 'darkred', width = barWidth,
         edgecolor = 'grey', label = 'Horticulture Workers')
plt.bar(br4, vegetablestotal, color = 'lightblue', width = barWidth,
         edgecolor = 'grey', label = 'Vegetable Workers')
plt.bar(br5, othercroptotal, color = 'grey', width = barWidth,
        edgecolor = 'grey', label = 'Other Crop Workers')



plt.xlabel('Crop Type', fontweight ='bold', fontsize = 15)
plt.ylabel('Obs. Each Year', fontweight ='bold', fontsize = 15)


plt.xticks([r + barWidth for r in range(len(fieldcroptotal))],
        ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'])

plt.legend()
plt.show()


##########################################################################

# Average Salary Graph by Position and Year
# Hourly Wage Adjusted for CPI after Tax

yearlydata = [naws2009, naws2010, naws2011, naws2012, naws2013, naws2014, naws2015, naws2016, naws2017, naws2018]


# Creating Datasets of Work and Year
field2009 = naws2009[naws2009['fieldwork']==1]
nursery2009 = naws2009[naws2009['nursery']==1]
packing2009 = naws2009[naws2009['packinghouse']==1]
other2009 = naws2009[naws2009['other']==1]

field2010 = naws2010[naws2010['fieldwork']==1]
nursery2010 = naws2010[naws2010['nursery']==1]
packing2010 = naws2010[naws2010['packinghouse']==1]
other2010 = naws2010[naws2010['other']==1]

field2011 = naws2011[naws2011['fieldwork']==1]
nursery2011 = naws2011[naws2011['nursery']==1]
packing2011 = naws2011[naws2011['packinghouse']==1]
other2011 = naws2011[naws2011['other']==1]

field2012 = naws2012[naws2012['fieldwork']==1]
nursery2012 = naws2012[naws2012['nursery']==1]
packing2012 = naws2012[naws2012['packinghouse']==1]
other2012 = naws2012[naws2012['other']==1]

field2013 = naws2013[naws2013['fieldwork']==1]
nursery2013 = naws2013[naws2013['nursery']==1]
packing2013 = naws2013[naws2013['packinghouse']==1]
other2013 = naws2013[naws2013['other']==1]

field2014 = naws2014[naws2014['fieldwork']==1]
nursery2014 = naws2014[naws2014['nursery']==1]
packing2014 = naws2014[naws2014['packinghouse']==1]
other2014 = naws2014[naws2014['other']==1]

field2015 = naws2015[naws2015['fieldwork']==1]
nursery2015 = naws2015[naws2015['nursery']==1]
packing2015 = naws2015[naws2015['packinghouse']==1]
other2015 = naws2015[naws2015['other']==1]

field2016 = naws2016[naws2016['fieldwork']==1]
nursery2016 = naws2016[naws2016['nursery']==1]
packing2016 = naws2016[naws2016['packinghouse']==1]
other2016 = naws2016[naws2016['other']==1]

field2017 = naws2017[naws2017['fieldwork']==1]
nursery2017 = naws2017[naws2017['nursery']==1]
packing2017 = naws2017[naws2017['packinghouse']==1]
other2017 = naws2017[naws2017['other']==1]

field2018 = naws2018[naws2018['fieldwork']==1]
nursery2018 = naws2018[naws2018['nursery']==1]
packing2018 = naws2018[naws2018['packinghouse']==1]
other2018 = naws2018[naws2018['other']==1]




avgsalary = [naws2009['wati'].mean(),naws2010['wati'].mean(),naws2011['wati'].mean(),naws2012['wati'].mean(),naws2013['wati'].mean(),naws2014['wati'].mean(),
               naws2015['wati'].mean(),naws2016['wati'].mean(),naws2017['wati'].mean(),naws2018['wati'].mean()]
fieldsalary = [field2009['wati'].mean(),field2010['wati'].mean(),field2011['wati'].mean(),field2012['wati'].mean(),field2013['wati'].mean(),field2014['wati'].mean(),
               field2015['wati'].mean(),field2016['wati'].mean(),field2017['wati'].mean(),field2018['wati'].mean()]
nurserysalary = [nursery2009['wati'].mean(),nursery2010['wati'].mean(),nursery2011['wati'].mean(),nursery2012['wati'].mean(),nursery2013['wati'].mean(),nursery2014['wati'].mean(),
               nursery2015['wati'].mean(),nursery2016['wati'].mean(),nursery2017['wati'].mean(),nursery2018['wati'].mean()]
packingsalary = [packing2009['wati'].mean(),packing2010['wati'].mean(),packing2011['wati'].mean(),packing2012['wati'].mean(),packing2013['wati'].mean(),packing2014['wati'].mean(),
               packing2015['wati'].mean(),packing2016['wati'].mean(),packing2017['wati'].mean(),packing2018['wati'].mean()]
othersalary = [other2009['wati'].mean(),other2010['wati'].mean(),other2011['wati'].mean(),other2012['wati'].mean(),other2013['wati'].mean(),other2014['wati'].mean(),
               other2015['wati'].mean(),other2016['wati'].mean(),other2017['wati'].mean(),other2018['wati'].mean()]


plt.plot(avgsalary, color = 'black', linestyle='-', label = 'Average Salary')
plt.plot(fieldsalary, color = 'darkgreen', linestyle='-', label = 'FieldWorker Avg Salary')
plt.plot(nurserysalary, color = 'lightblue', linestyle='-', label = 'Nursery Avg Salary')
plt.plot(packingsalary, color = 'brown', linestyle='-', label = 'Packing Avg Salary')
plt.plot(othersalary, color = 'darkred',linestyle='-', label = 'Other Workers Avg Salary')
plt.title('Average Salary by Worktype',fontweight ='bold', fontsize = 15)
plt.xlabel('Year',fontweight ='bold', fontsize = 10)
plt.ylabel('Average Salary in Dollars',fontweight ='bold', fontsize = 10)
plt.legend(loc='upper left')
plt.show()


plt.plot(avgsalary, color = 'black', linestyle='-', label = 'Average Salary')
plt.plot(fieldsalary, color = 'darkgreen', linestyle='-', label = 'FieldWorker Avg Salary')
plt.plot(nurserysalary, color = 'lightblue', linestyle='-', label = 'Nursery Avg Salary')
plt.plot(packingsalary, color = 'brown', linestyle='-', label = 'Packing Avg Salary')
plt.title('Average Salary by Worktype',fontweight ='bold', fontsize = 15)
plt.xlabel('Year',fontweight ='bold', fontsize = 10)
plt.ylabel('Average Salary in Dollars',fontweight ='bold', fontsize = 10)
plt.legend(loc='upper left')
plt.show()

###############################################################################################3






easttotal = [naws2009['east'].sum(), naws2010['east'].sum(), naws2011['east'].sum(), naws2012['east'].sum(), naws2013['east'].sum(), naws2014['east'].sum(),
             naws2015['east'].sum(), naws2016['east'].sum(), naws2017['east'].sum(), naws2018['east'].sum()]
southeasttotal = [naws2009['southeast'].sum(), naws2010['southeast'].sum(), naws2011['southeast'].sum(), naws2012['southeast'].sum(), naws2013['southeast'].sum(), naws2014['southeast'].sum(),
             naws2015['southeast'].sum(), naws2016['southeast'].sum(), naws2017['southeast'].sum(), naws2018['southeast'].sum()]
midwesttotal = [naws2009['midwest'].sum(), naws2010['midwest'].sum(), naws2011['midwest'].sum(), naws2012['midwest'].sum(), naws2013['midwest'].sum(), naws2014['midwest'].sum(),
             naws2015['midwest'].sum(), naws2016['midwest'].sum(), naws2017['midwest'].sum(), naws2018['midwest'].sum()]
southwesttotal = [naws2009['southwest'].sum(), naws2010['southwest'].sum(), naws2011['southwest'].sum(), naws2012['southwest'].sum(), naws2013['southwest'].sum(), naws2014['southwest'].sum(),
             naws2015['southwest'].sum(), naws2016['southwest'].sum(), naws2017['southwest'].sum(), naws2018['southwest'].sum()]
northwesttotal = [naws2009['northwest'].sum(), naws2010['northwest'].sum(), naws2011['northwest'].sum(), naws2012['northwest'].sum(), naws2013['northwest'].sum(), naws2014['northwest'].sum(),
             naws2015['northwest'].sum(), naws2016['northwest'].sum(), naws2017['northwest'].sum(), naws2018['northwest'].sum()]
californiatotal = [naws2009['california'].sum(), naws2010['california'].sum(), naws2011['california'].sum(), naws2012['california'].sum(), naws2013['california'].sum(), naws2014['california'].sum(),
             naws2015['california'].sum(), naws2016['california'].sum(), naws2017['california'].sum(), naws2018['california'].sum()]


barWidth = 0.20
br1 = np.arange(len(fieldcroptotal))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]


plt.bar(br1, easttotal, color = 'lightblue', width = barWidth,
        edgecolor = 'grey', label = 'East')
plt.bar(br2, southeasttotal, color = 'rebeccapurple', width = barWidth,
        edgecolor = 'grey', label = 'Southeast')
plt.bar(br3, midwesttotal, color = 'darkred', width = barWidth,
        edgecolor = 'grey', label = 'Midwest')
plt.bar(br4, southwesttotal, color = 'yellow', width = barWidth,
        edgecolor = 'grey', label = 'Southwest')
plt.bar(br5, northwesttotal, color = 'darkgreen', width = barWidth,
        edgecolor = 'grey', label = 'Northwestt')
plt.bar(br6, californiatotal, color = 'orange', width = barWidth,
        edgecolor = 'grey', label = 'California')

plt.xlabel('Region', fontweight ='bold', fontsize = 15)
plt.ylabel('Obs. Each Year', fontweight ='bold', fontsize = 15)


plt.xticks([r + barWidth for r in range(len(easttotal))],
        ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'])

plt.legend()
plt.show()












