# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 23:56:09 2017

@author: Patrick
"""

import matplotlib.pyplot as plt
import numpy as np

dp = np.array([[3,4], [5,4], [4,1], [4,2], [3,2], [2,3]])

dp = np.transpose(dp)

#img 1
plt.figure(figsize = (8,8))
plt.scatter(dp[0], dp[1], s= 40)
plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
plt.title(r"$\epsilon = 0$")

#img 2 radius 1
dp_2 = np.transpose(np.array([[4,2], [4,1]]))
dp_3 = np.transpose(np.array([[4,2], [3,2]]))
plt.figure(figsize = (8,8))
plt.plot(dp_2[0], dp_2[1], color = 'black')
plt.plot(dp_3[0], dp_3[1], color = 'black')
plt.scatter(dp[0], dp[1], s= 40)


plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
area = 100*4*4*4*1.3
plt.scatter(dp[0], dp[1], s= area, alpha = 0.5, color = "pink")

plt.title(r"$\epsilon = 1$")

#img 3 radius sqrt(2) = 1.41
dp_4 = np.transpose(np.array([[2,3], [4,1]]))
dp_5 = np.transpose(np.array([[2,3], [3,4]]))

plt.figure(figsize = (8,8))
plt.plot(dp_2[0], dp_2[1], color = 'black')
plt.plot(dp_3[0], dp_3[1], color = 'black')
plt.plot(dp_4[0], dp_4[1], color = 'black')
plt.plot(dp_5[0], dp_5[1], color = 'black')

plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
area = 100*4*4*4*2.5
plt.scatter(dp[0], dp[1], s= area, alpha = 0.5, color = "pink")

x = np.array([3,4,4,3])
y = np.array([2,2,1,2])
plt.fill_betweenx(y,x, alpha = 0.5)

plt.scatter(dp[0], dp[1], s= 40, color = "C0")
plt.title(r"$\epsilon = \sqrt{2}$")

#img 4 radius 2 
dp_6 = np.transpose(np.array([[3,4],[3,2]]))
dp_7 = np.transpose(np.array([[3,4],[5,4]]))
plt.figure(figsize = (8,8))
plt.plot(dp_2[0], dp_2[1], color = 'black')
plt.plot(dp_3[0], dp_3[1], color = 'black')
plt.plot(dp_4[0], dp_4[1], color = 'black')
plt.plot(dp_5[0], dp_5[1], color = 'black')
plt.plot(dp_6[0], dp_6[1], color = 'black')
plt.plot(dp_7[0], dp_7[1], color = 'black')

plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
area = 100*4*4*4*4*1.28
plt.scatter(dp[0], dp[1], s= area, alpha = 0.5, color = "pink")

x = np.array([3,4,4,3,2,3,3])
y = np.array([2,2,1,2,3,4,2])
plt.fill_betweenx(y,x, alpha = 0.5)

plt.scatter(dp[0], dp[1], s= 40, color = "C0")

plt.title(r"$\epsilon = 2$")


# img 5 radius sqrt(5) = 2.236
dp_8 = np.transpose(np.array([[3,4],[4,2]]))
dp_9 = np.transpose(np.array([[4,2],[5,4]]))
plt.figure(figsize = (8,8))
plt.plot(dp_2[0], dp_2[1], color = 'black')
plt.plot(dp_3[0], dp_3[1], color = 'black')
plt.plot(dp_4[0], dp_4[1], color = 'black')
plt.plot(dp_5[0], dp_5[1], color = 'black')
plt.plot(dp_6[0], dp_6[1], color = 'black')
plt.plot(dp_7[0], dp_7[1], color = 'black')
plt.plot(dp_8[0], dp_8[1], color = 'black')
plt.plot(dp_9[0], dp_9[1], color = 'black')
#plt.scatter(dp[0], dp[1], s= 40)

plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
area = 100*4*4*4*4*1.5
plt.scatter(dp[0], dp[1], s= area, alpha = 0.5, color = "pink")

x = np.array([4,2,3,5,4,4])
y = np.array([1,3,4,4,2,1])
plt.fill_betweenx(y,x, alpha = 0.5)

plt.scatter(dp[0], dp[1], s= 40, color = "C0")

plt.title(r"$\epsilon = \sqrt{5}$")

# img 6 radius sqrt(10) = 3.16
dp_10 = np.transpose(np.array([[4,1],[5,4]]))
plt.figure(figsize = (8,8))
plt.plot(dp_2[0], dp_2[1], color = 'black')
plt.plot(dp_3[0], dp_3[1], color = 'black')
plt.plot(dp_4[0], dp_4[1], color = 'black')
plt.plot(dp_5[0], dp_5[1], color = 'black')
plt.plot(dp_6[0], dp_6[1], color = 'black')
plt.plot(dp_7[0], dp_7[1], color = 'black')
plt.plot(dp_8[0], dp_8[1], color = 'black')
plt.plot(dp_9[0], dp_9[1], color = 'black')
plt.plot(dp_10[0], dp_10[1], color = 'black')

plt.ylim(0.5, 5.5)
plt.xlim(0.5, 5.5)
area = 100*4*4*4*4*3
plt.scatter(dp[0], dp[1], s= area, alpha = 0.5, color = "pink")

x = np.array([4,2,3,5,4])
y = np.array([1,3,4,4,1])
plt.fill_betweenx(y,x, alpha = 0.5)

plt.scatter(dp[0], dp[1], s= 40, color = "C0")


plt.title(r"$\epsilon = \sqrt{10}$")