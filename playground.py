import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(3,-3,100)

def step(x,N):
	sum = 0
	for n in range(1,N+1):
		sum += (-1)**n/(np.pi*n)*np.sin(2*n*np.pi*x)

	return x + sum
#y = errf(x)
y = step(x,1000)

plt.plot(x,y)
plt.show()
