import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(-3,3,100)
#y = x - np.sin(2*np.pi*x)/(2*np.pi)
#plt.plot(x,y)
#plt.show()

np.random.seed(42)

N = 2
T = 1000
lam1 = 0.001
lam2 = 0.001
alpha = 0.01
#w = np.random.rand(N**2)
w = np.asarray([1,2,2,3]).reshape((-1,1))
m = np.asarray([1,-1]).reshape((-1,1)) #Optimal: [2,-1], A: [1 2,1 0,1 0,2 1]
A = (2*np.random.rand(N**2,2)).astype(np.int)

def rounding_func(A):
	return A - np.sin(2*np.pi*A)/(2*np.pi)

def get_gradient_A(lam1,A,m,w):
	t0 = 2*np.pi
	T1 = t0*A
	t2 = np.matmul(A-1/t0*np.sin(T1),m)-w
	grad = np.matmul(t2,m.T)-t2 #np.diagflat(t2)
	return grad

def get_gradient_m(lam1,A,w):
	t0 = 2*np.pi
	T1 = A-1/t0 * np.sin(t0*A)
	grad = np.matmul(T1.T, np.matmul(T1,m)-w)
	return grad

def get_loss(lam1,A,m,w):
	return lam1/2*np.linalg.norm(np.matmul(rounding_func(A),m)-w,2)**2

for i in range(T):
	loss = get_loss(lam1,A,m,w)
	if(i % 2 == 0):
		m = m - alpha*get_gradient_m(lam1,A,w)
	else:
		A = A - alpha*get_gradient_A(lam1,A,m,w)
	print(loss)

print(A)
print(rounding_func(A))
print(m)
print(w)
print(np.matmul(rounding_func(A),m))
