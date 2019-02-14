import numpy as np
import matplotlib.pyplot as plt
import math
import subprocess
import shlex

n=np.array([1,-1])  #normal vector
omat=np.array([[0,1],[-1,0]])
m=np.matmul(omat,n)   #direction of given line
#print(m)
A=np.array([2,1])   #given point

B=2*math.sqrt(3)*m/math.sqrt(m[0]**2+m[1]**2)+A   #Required point

C=np.array([4,0])    #point on given line

lamda=np.linspace(-5,4,40)
l1=np.zeros((2,40))
l2=np.zeros((2,40))

for i in range(40):
	temp1=C+lamda[i]*m
	l1[:,i]=temp1.T	
for i in range(40):
    temp2=B+lamda[i]*n	
    l2[:,i]=temp2.T

plt.plot(l1[0,:],l1[1,:],label='$L_1:x-y=4$')
plt.plot(l2[0,:],l2[1,:],label='$L_2:x+y=3-2\sqrt{6}$')
plt.plot(2,1,'o')
plt.text(2*(1+0.1),1+0.1,'A')
plt.text(B[0]*(1+0.1),B[1]*(1-0.2),'B')
plt.plot(B[0],B[1],'*')
plt.legend(loc='best')
plt.grid()

plt.savefig('/home/binaya/Desktop/AI & ML/figure.eps')
plt.savefig('/home/binaya/Desktop/AI & ML/figure.pdf')
plt.show()
print(B)

