pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

import numpy as np
data=np.genfromtxt("iris.data", delimiter=',')
print(data)

print("data type:", type(data))
print("data shape:", data.shape)
print(data[:10])

data1= np.genfromtxt("iris.data", delimiter=",", dtype=None)
print(data1.shape)
print(type(data1))
print(type(data1[0]))
print(type(data1[0][4]))
print(data1[:10])

dt=np.dtype("f8, f8, f8, f8, U30")
data2= np.genfromtxt("iris.data", delimiter=",", dtype=dt)
print(data2.shape)
print(type(data2))
print(type(data2[0]))
print(type(data2[0][4]))
print(data2[:10])

import matplotlib as mpl
import matplotlip.pyplot as plt

#Данные из отдельных столбцов
sepal_length=[] # sepal lenght
sepal_width=[] # sepal widht
petal_length=[] # sepal lenght
petal_width=[] # sepal wight

#выполняем обход всей коллекции data2
for dot in data2
    sepal_length.append(dot[0])
    sepal_width.append(dot[1])
    petal_length.append(dot[2])
    petal_width.append(dot[3])

#строим графики по проекциям данных

plt.figure(1)
setosa, = plt.plot(sepal_length[:50], sepal_width[:50],'ro', label='setosa' )
versicolor, = plt.plot(sepal_length[50:100], sepal_width[50:100],'g^', label='versicolor')
virginica, = plt.plot(sepal_length[100:150], sepal_width[100:150],'bs', label='verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('sepal length')
plt.ylabel('sepal width')

plt.figure(2)
setosa, = plt.plot(sepal_length[:50], petal_width[:50],'ro', label='setosa' )
versicolor, = plt.plot(sepal_length[50:100], petal_width[50:100],'g^', label='versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_width[100:150],'bs', label='verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('sepal length')
plt.ylabel('petal length')

plt.figure(3)
setosa, = plt.plot(sepal_length[:50], petal_width[:50],'ro', label='setosa' )
versicolor, = plt.plot(sepal_length[50:100], petal_width[50:100],'g^', label='versicolor')
virginica, = plt.plot(sepal_length[100:150], petal_width[100:150],'bs', label='verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('sepal length')
plt.ylabel('petal width')

plt.show()