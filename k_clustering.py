

from matplotlib import category
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from random import sample
import string

#Read data files 
animals = pd.read_csv("animals", sep = " ", header = None, encoding = 'unicode_escape')
countries = pd.read_csv( "countries", sep = " ", header = None)
fruits = pd.read_csv( "fruits", sep = " ", header = None)
veggies = pd.read_csv( "veggies", sep = " ", header = None)

#Add a new column to keep the category of data
#0 for animals
#1 for veggies
#2 for fruits
#3 for countries
animals['Category'] = '0'
countries['Category']= '1'
fruits['Category'] = '2'
veggies['Category'] = '3'

#Merging the datas to a single dataframe
data = pd.concat([animals, countries, fruits, veggies ])

#Creating a table containing names of elements and the respective category
new_data = (pd.concat([data.iloc[:, :1],data.iloc[:,-1:]],axis = 1)).values.tolist()

#Initialise four empty list ani, coun, fru, veg for storing 
#each category seperately for Question 3 to 6
ani = []
coun = []
fru = []
veg =[]
for ele in new_data:
  if ele[1] == '0':
    ani.append(ele[0])
    len(ani)+1
  elif ele[1] == '1':
    coun.append(ele[0])
  elif ele[1] == '2':
    fru.append(ele[0])
  else:
    veg.append(ele[0])

#Merging the above lists together and produce a single list
#for referring the index value for Bcubed calculation
name_list = ani + coun + fru + veg

#Extracting the first column of names of contents and storing as a list
first_column = data.iloc[: , :1].values.tolist()

#Deleting the column containing the category of data
data.drop('Category',inplace = True, axis = 1)


#Select the features
featur = data.select_dtypes(include='number')
featur_data = featur.to_numpy()




def k_clustering(X, k, distance, normalisation):

    """
    Function k_clustering can be used to cluster the data points
    using the k means and k medians algorithm.

    Parameters:

      X numpy.ndarray: input dataset
      k integer: No:of clusters
      distance string: 
        for k_means- distance = "Euclidean"
        for k_median- distance = "Manhattan"
      normalisation boolean:
        if True : The data set X will be normalised 

    Returns:

      final_dict dictionary:
        Keys: Clusters(ranging from A to I depending on the k value)
        Values : Names of each object given in the dataset
       
    """
    #features are normalised for question no: 4 and 6
    if normalisation == True:      
      
      X = X/ np.linalg.norm(X, axis = 1).reshape(-1,1)

    #Initialize list and dictionaries 
    cluster = []
    m = []
    final_dict= {}
    centroids = []

    np.random.seed(1)

    #Find initial centroid
    idx = np.random.randint(X.shape[0], size = k)
    cluster.append(idx)
    for i in idx:   
      w= X[i]      
      centroids.append(w)
 
    # Create copies of the centroids for updating
    centroids_new = deepcopy(centroids)
    centroids_old = np.zeros(np.shape(centroids))

    #Create names of clusters as uppercase alphabets
    alpha = list(string.ascii_uppercase)
    m = alpha[0:len(centroids_new)]
    
    #calculate difference between existing centroid and new centroid
    variation = np.subtract(centroids_new, centroids_old)
    number_of_iterations = 0

    #when the new centroid points is not equal to existing points
    while not np.all((variation == 0)):

        #update number of iterations
        number_of_iterations += 1
        print("number of iteration=", number_of_iterations)

        #Initialise an array for storing distance between centroids and datapoints
        dist = np.zeros([featur_data.shape[0], k])
        c_dict = {}
        Clust = {}
        C =[]
        y = []

        #calculating new centroids for kmeans algorithm
        if distance == "Euclidean":

            #Calculate Euclidean distance
            for j in range(len(centroids_new)):
              dist[:,j]= np.linalg.norm((X - centroids_new[j]), axis =1)
            
            #Finding the minimum distance
            x = np.argmin(dist, axis=1)   

            #Assigning cluster to each data point
            for i in x:              
              Clusters = m[i]                
              C.append(Clusters)

            #Creating dictionary which store all the data points as values to the key cluster
            for key, value in zip(C, X):
              c_dict.setdefault(key, []).append(value)
            Clust.update(c_dict)

            #Assign the new copy of centroids to the old centroids object
            centroids_old = deepcopy(centroids_new)  

            #Find the mean to update new centroids          
            for key,value in Clust.items():        
                p = np.mean(value, axis = 0)
                y.append(p)
            centroids_new = y
            
        #calculating new centroids for kmedians algorithm  
        elif distance == "Manhattan":

            #Calculate Manhattan distance
            for j in range(len(centroids_new)):
                dist[:, j] = np.sum(np.abs(X - centroids_new[j]), axis=1)

            #Finding the minimum distance    
            x = np.argmin(dist, axis=1)   

            #Assigning cluster to each data point
            for i in x:
              Clusters = m[i]   
              C.append(Clusters)

            #Creating dictionary which store all the data points as values to the key cluster  
            for key, value in zip(C, X):
              c_dict.setdefault(key, []).append(value)
            Clust.update(c_dict)

            #Assign the new copy of centroids to the old centroids object
            centroids_old = deepcopy(centroids_new)

            #Find the mean to update new centroids 
            for key,value in Clust.items():
              p = np.median(value, axis = 0)
              y.append(p)
            centroids_new = y
   
      # Re-calculate the variation between centroids  
        variation = np.subtract(centroids_new, centroids_old)


    #Creating a final dictionary to store clusters and datapoints converged to each cluster    
    for key, value in zip(C, first_column):
        final_dict.setdefault(key, []).append(value)

    #Printing the Clusters
    for key, value in final_dict.items():
        print("Number of elements in the Cluster =",len(final_dict[key]))
        print (key, '\n\n\n', value)
        print("\n")
      

    return final_dict
      
#####################################################################################################
#Question 1

#k_clustering(featur_data, 4, "Euclidean", False)

##################################################################################################### 

#Question 2

#k_clustering(featur_data, 4, "Manhattan", False)

#####################################################################################################

"""
#Question 3 and Question 4
#Initialise list
k_list = []
precision = []
recall = []
f_score = []

#Iterating k value from 1 to 9
for k in range(1,10):

  #Store k value in a list for plotting
  k_list.append(k)

  #Normalisation is set to false for Question 3
  #final_dict = k_clustering(featur_data, k, "Euclidean", False)

  #Normalisation is set to True for Question 5
  #final_dict = k_clustering(featur_data, k, "Euclidean", True)

  #initialize an array for storing precision recall and fscore for each element
  b_cubed = np.zeros([featur_data.shape[0], 3])


  for key , value in final_dict.items():

    #Convert the value into a list   
    value_list = [item for sublist in value for item in sublist]

    #for each element calculate precision, recall and f_score
    for element in value_list:
      if element in ani:
        num = len(set(ani).intersection(set(value_list)))
        #precision
        b_cubed[name_list.index(element),0] = (num/len(value_list)) 
        #recall
        b_cubed[name_list.index(element),1] = (num/len(ani)) 
        #f_score
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])
        
      
      elif element in coun:
        num = len(set(coun).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(coun))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

      elif element in fru:
        num = len(set(fru).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(fru))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

      elif element in veg:
        num = len(set(veg).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(veg))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

  #calculate the average precision , recall and f_score
  average = b_cubed.mean(axis =0)

  #create 3 list containg Precision, recall and f_score for various k values
  precision.append(average[0])
  recall.append(average[1])
  f_score.append(average[2])
  print("Precision is ", [ round(elem, 4) for elem in precision ])
  print("recall is :",[ round(elem, 4) for elem in recall ])
  print("f_score is :", [ round(elem, 4) for elem in f_score ])


# Plot K_list against Precision
plt.plot(k_list, precision, label="Precision")
# Plot K_list against Recall
plt.plot(k_list, recall, label="Recall")
# Plot K_list against F_score
plt.plot(k_list, f_score, label="F-Score")
# Plot the title
#Title for Question : 3
#plt.title("K-Means Clustering " )
#Title for Question 4
#plt.title("K-Means Clustering with normalised features" )
# Plot the x and y axis labels
plt.xlabel('Number of Clusters')
plt.ylabel("Bcubed Scores")
# Display the legend
plt.legend()
# Display the plot
plt.show()    

#"""

#####################################################################################################

"""
#Question 5 and Question 6 
#Initialise list
k_list = []
precision = []
recall = []
f_score = []

#Iterating k value from 1 to 9
for k in range(1,10):

  #Store k value in a list for plotting
  k_list.append(k)
  
  #Normalisation is set to false for Question 4
  #final_dict = k_clustering(featur_data, k, "Manhattan", False)

  #Normalisation is set to true for Question 6
  final_dict = k_clustering(featur_data, k, "Manhattan", True)

  #initialize an array for storing precision recall and fscore for each element
  b_cubed = np.zeros([featur_data.shape[0], 3])

  for key , value in final_dict.items():

    #Convert the value into a list  
    value_list = [item for sublist in value for item in sublist]

    #for each element calculate precision, recall and f_score
    for element in value_list:
      if element in ani:
        num = len(set(ani).intersection(set(value_list)))
        #precision
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        #recall
        b_cubed[name_list.index(element),1] = (num/len(ani))
        #f_score
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])
        
      
      elif element in coun:
        num = len(set(coun).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(coun))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

      elif element in fru:
        num = len(set(fru).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(fru))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

      elif element in veg:
        num = len(set(veg).intersection(set(value_list)))
        b_cubed[name_list.index(element),0] = (num/len(value_list))
        b_cubed[name_list.index(element),1] = (num/len(veg))
        b_cubed[name_list.index(element),2] = (2 * b_cubed[name_list.index(element),0] * b_cubed[name_list.index(element),1] )/(b_cubed[name_list.index(element),0] + b_cubed[name_list.index(element),1])

  #calculate the average precision , recall and f_score
  average = b_cubed.mean(axis =0)
  
  #create 3 list containg Precision, recall and f_score for various k values
  precision.append(average[0])
  recall.append(average[1])
  f_score.append(average[2])
  print("Precision is ", [ round(elem, 4) for elem in precision ])
  print("recall is :",[ round(elem, 4) for elem in recall ])
  print("f_score is :", [ round(elem, 4) for elem in f_score ])


# Plot K_list against Precision
plt.plot(k_list, precision, label="Precision")
# Plot K_list against Recall
plt.plot(k_list, recall, label="Recall")
# Plot K_list against F_score
plt.plot(k_list, f_score, label="F-Score")
# Plot the title
#Title for question 5
#plt.title("K-Medians Clustering " )
#Title for Question 6
plt.title("K-Medians Clustering with normalised features" )
# Plot the x and y axis labels
plt.xlabel('Number of Clusters')
plt.ylabel("Bcubed Score")
# Display the legend
plt.legend()
# Display the plot
plt.show()    

#"""

    
  

