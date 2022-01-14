# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### Load in Numpy

import numpy as np

### The Basics

x = np.array([1, 2, 3])
print(x)

y = np.array([[9.0, 8.0, 7.0], [9.0, 8.0, 7.0]])
print(y)

### Get Dimensions

print(x.ndim)
print(y.ndim)

### Get Shape

print(y.shape)

### Get Type

print(y.dtype)
print(x.dtype)

### Accessing/Changing Specific elements, ros, columns, etc.

x = np.array([[1,2,3,4,5,6,7,8],
              [18,17,16,15,14,13,12,31]])

# Numpy Array Strat with index 0

print(x[1,3])

print(x[1,-2])

### Get a specific row/column

print(x[0,:]) # row

print(x[:,2]) # column

### Get a little more fancy [start_idx:end_idx:step_size]

print(x[1,1:10:3])

### Change Specific Index Value

x[1,5] = 100

print(x)

x[1,2:6] = 50
print(x)

### Initialize Different types of Arrays

# All 0s Matrix

z = np.zeros((3,2))
print(z)

# All 1s Matrix

z = np.ones((4,2))
print(z)

# All any number Matrix
z = np.full((3,10), 120)
print(z)

# Any other number (full_like), copy the same size of the other array, but change all the values to the specific new value

z2 = np.full_like(z, 520)
print(z2)

### Identity Matrix

x = np.identity(5)
print(x)

# Random decimal numbers

x = np.random.rand(4,2)
print(x)

# Repeat an array

arr = np.array([[123,223,333]])
r1 = np.repeat(arr,3, axis=0)
print(r1)

arr = np.array([[123,223,333]])
r1 = np.repeat(arr,3, axis=1)
print(r1)

### Copy an Array, do not use (=). You need to use copy() Method

a = np.array([1,2,3])
b = a.copy()
b[0] = 100

print('a=\n',a)
print('b=\n',b)

### Load Data from File

# filedata = np.genfromtxt('data.txt', delimiter=',')
# filedata = filedata.astype('int32')
# print(filedata)

### Reorganizing Arrays

x = np.array([[1,2,3,4,5,6,7,8],
              [18,17,16,15,14,13,12,31]])

y = np.reshape(x, (8,2))
print(y)

# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

v3 = np.vstack([v1,v2,v1,v2])
print(v3)
# Horizontal  stack
h1 = np.ones((2,4))
h2 = np.zeros((2,2))

h3 = np.hstack((h1,h2))
print(h3)
