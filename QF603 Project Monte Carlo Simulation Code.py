#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Modify mean and covariance variables here

mean_returns=Data.mean()
cov_matrix= Data.cov()

#Set the number of iterations to 100000 (I set to 200 just to test). Change "Data.columns" to suit your code
num_iterations = 200
simulation_res = np.zeros((4+len(Data.columns)-1,num_iterations))

for i in range(num_iterations):
#Select random weights and normalize to set the sum to 1 (change number inside "rand()", should be number of stocks in covariance)
        weights = np.array(np.random.rand(100))
        weights /= np.sum(weights)
        
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
indexes = pd.read_csv('indexes.csv', index_col=0)
simulation.index = indexes.index

#Print and export this to get the Returns, Stdev, Sharpe and Weights of all iterations
print(pd.DataFrame(simulation))
simulation.to_csv('simulation_results.csv')

sim_frame = pd.DataFrame(simulation_res.T)
ret = sim_frame[0]
stdev = sim_frame[1]

#Create a scatter plot coloured by various Sharpe Ratios with standard deviation on the x-axis and returns on the y-axis
plt.figure(figsize=(10,10))
plt.scatter(stdev,ret,s=1,c='b')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')

