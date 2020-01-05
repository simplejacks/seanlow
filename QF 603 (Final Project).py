#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import os
import glob

temp = pd.read_excel(os.path.expanduser("/Users/seanlow/Desktop/QF603/603 Project/Project_603.xlsx"), index_col=0)
sample = temp[temp.Team==6]; #print(sample)

path = '/Users/seanlow/Desktop/QF603/603 Project/2018'
os.chdir(path)

Data = pd.DataFrame()

for i in sample.index:
    temp = pd.read_csv(i, header=0, index_col=0,parse_dates=True)
    Data[i] = temp[temp.index>'1999-04-01']["Adj Close"].resample('M').agg({'Adj Close':'last'}).pct_change()

Data=Data.drop(Data.index[0])


# In[161]:


path = '/Users/seanlow/Desktop/QF603/603 Project'
os.chdir(path)
FF_factors = pd.read_csv('FF.csv', header=0, index_col=0)
SNP500 = pd.read_csv('^GSPC.csv', header=0, index_col=0, parse_dates=True)


# In[162]:


Rf = np.array([FF_factors.iloc[:,3].values]).T
Ri_Rf = Data-Rf
Rm = SNP500[SNP500.index>'1999-04-01']["Adj Close"].resample('M').agg({'Adj Close':'last'}).pct_change()
Rm = pd.DataFrame(Rm.drop(Rm.index[0]))
Rm_Rf = Rm-Rf
Rm_Rf_df = pd.DataFrame(np.repeat(Rm_Rf.values,100,axis=1))

Regression = LinearRegression()
Regression.fit(Rm_Rf_df,Ri_Rf)

beta=pd.DataFrame(Regression.coef_)
alpha = pd.DataFrame(Regression.intercept_, columns=['alphas'])
alpha.index=Ri_Rf.columns

sort_alpha = alpha.sort_values('alphas', ascending=False)
sort_alpha.to_csv('/Users/seanlow/Desktop/QF603/603 Project/Sort_alphas.csv')

select = np.append(sort_alpha.index[0:10].values, sort_alpha.index[90:].values)

selected = Data[select]

#pd.DataFrame(selected.columns).to_csv('selected.csv')


# In[169]:


#Modify mean and covariance variables here

mean_returns=selected.mean()
cov_matrix = selected.cov()

#Set the number of iterations to 100000 (I set to 200 just to test). Change "Data.columns" to suit your code
num_iterations = 100000
simulation_res = np.zeros((4+len(selected.columns)-1,num_iterations))

for i in range(num_iterations):
#Select random weights and normalize to set the sum to 1 (change number inside "rand()", should be number of stocks in covariance)
        wp = np.array(np.random.rand(10))
        weightspositive = 1/wp
        weightspositive /= np.sum(weightspositive)
        
        wn = np.array(np.random.rand(10))
        weightsnegative = 1/wn
        weightsnegative /= -np.sum(weightsnegative)
        
        weights = np.concatenate((weightspositive, weightsnegative))
        
        
#Calculate the return and standard deviation for every step
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
        
#Store all the results in a defined array
        simulation_res[0,i] = portfolio_return
        simulation_res[1,i] = portfolio_std_dev


#Calculate Sharpe ratio and store it in the array
        simulation_res[2,i] = simulation_res[0,i] / simulation_res[1,i]
    
       #Save the weights in the array
        for j in range(len(weights)):
              simulation_res[j+3,i] = weights[j]
                
simulation = pd.DataFrame(simulation_res)
indexes = pd.read_csv('selected.csv', index_col=0)
simulation.index = indexes.index

#Print and export this to get the Returns, Stdev, Sharpe and Weights of all iterations
#print(pd.DataFrame(simulation.T))
sort_sharpe = (simulation.T).sort_values('Sharpe', ascending=False)
print(sort_sharpe)
sort_sharpe.to_csv('simulation_results.csv')

sim_frame = pd.DataFrame(simulation_res.T)
ret = sim_frame[0]
stdev = sim_frame[1]

#Create a scatter plot coloured by various Sharpe Ratios with standard deviation on the x-axis and returns on the y-axis
plt.figure(figsize=(10,10))
plt.scatter(stdev,ret,s=1,c='b')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')


# In[154]:


#####SIMULATION (TO BE MODIFIED)#######


x0=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1])
port = Ri_Rf[select]
portcov = selected.cov()

#####Changing path here to access my files#####
path = '/Users/seanlow/Desktop/QF603/603 Project/2018'
os.chdir(path)
######################################

R_mean = port.mean()
Portfolio_entry_prices=[]
for i in select:
    temp1=pd.read_csv(i,index_col=0,parse_dates=True)
    Portfolio_entry_prices.append(np.asscalar(temp1[temp1.index=='2018-12-31']["Adj Close"].values))
Long_Short=pd.DataFrame({'Portfolio Return':R_mean,'Price':Portfolio_entry_prices},index=select)
returns=np.array(Long_Short['Portfolio Return'])
#print(Long_Short)

long_prices=np.array(Long_Short['Price'][0:10])
short_prices=np.array(Long_Short['Price'][10:20])


#Our function
def fun(x,returns):
    return -(np.dot(x,returns.T))
    #return -(np.dot(x[0:10],returns[0:10].T))+(np.dot(x[10:20],returns[10:20].T))

def portvar(x,cov):
    return (x@cov@x.T)

def sharpe(x,returns,cov):
    return (fun(x,returns)/np.sqrt(portvar(x,cov)))

#def sortino(x,returns,cov):
#    r = Ri_Rf.copy(); r[r>0] = 0
#    semi=(r**2).mean()

#sigma = portvar(x0,portcov)
#print(sigma)

#Our conditions
cond = ({'type': 'eq', 'fun': lambda x: (np.dot(x[0:10],long_prices.T))-(np.dot(x[10:20],(short_prices.T)))},
        {'type': 'ineq', 'fun': lambda x: sum(x[:10])-1.0},
        {'type': 'eq', 'fun': lambda x: sum(x[10:])-1.0})
bnds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
        (-1,0),(-1,0),(-1,0),(-1,0),(-1,0),(-1,0),(-1,0),(-1,0),(-1,0),(-1,0))
result = optimize.minimize(sharpe,x0,args=(returns,portcov), bounds=bnds, constraints = cond)
#print(result.x)
print(sum(result.x*returns))
#print(sharpe(result.x,))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




