import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(-3,3,100)
#y = x - np.sin(2*np.pi*x)/(2*np.pi)
#plt.plot(x,y)
#plt.show()

#np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)

N = 2
N_sum = 100
T = 1000
lam1 = 1
lam2 = 0.001
alpha = 0.001
#w = np.random.rand(N**2)
w = np.asarray([1,2,2,3]).reshape((-1,1))
w = np.random.randn(N**2).reshape((-1,1))
#m = np.asarray([np.max(w),np.min(w)]).reshape((-1,1)) #Optimal: [2,-1], A: [1 2,1 0,1 0,2 1]
m = np.random.randn(2).reshape((-1,1))
#A = (2*np.random.rand(N**2,2)).astype(np.int)
A = np.abs(np.random.randn(N**2,2))

def round_sum(x,N):
	sum = 0
	for n in range(1,N+1):
		sum += (-1)**n/(np.pi*n)*np.sin(2*n*np.pi*x)
	return x + sum


def rounding_func(A):
	return A - np.sin(2*np.pi*A)/(2*np.pi)

def get_gradient_A(lam1,A,m,w):
	t0 = 2*np.pi
	T1 = t0*A
	t2 = np.matmul(A-1/t0*np.sin(T1),m)-w
	grad = np.matmul(t2,m.T)-t2 #np.diagflat(t2)
	return grad

def get_gradient_A_sum(lam1,A,m,w,N):
	t0 = np.zeros(A.shape)
	for n in range(1,N+1):
		t0 += (-1)**n/(np.pi*n)*np.sin(2*np.pi*A)
	t5 = np.matmul(A + t0, m) - w
	t2 = np.zeros(A.shape)
	for n in range(1,N+1):
		t2 += 2*(-1)**n*np.matmul(np.matmul(np.diagflat(t5),np.cos(2*n*np.pi*A)),np.diagflat(m))

	grad = np.matmul(t5,m.T) + t2
	return grad

def get_gradient_m_sum(lam1,A,w,N):
	T0 = np.zeros(A.shape)
	for n in range(1,N):
		T0 += (-1)**n/(np.pi*n)*np.sin(1*np.pi*n*A)

	T1 = A + T0
	grad = np.matmul(T1.T, np.matmul(T1,m)-w)
	return grad

def get_gradient_m(lam1,A,w):
	t0 = 2*np.pi
	T1 = A-1/t0 * np.sin(t0*A)
	grad = np.matmul(T1.T, np.matmul(T1,m)-w)
	return grad

def get_loss(lam1,A,m,w):
	return lam1/2*np.linalg.norm(np.matmul(round_sum(A,100),m)-w,2)**2


for i in range(T):
	loss = get_loss(lam1,A,m,w)
	if(i % 2 == 0):
		m = m - alpha*get_gradient_m_sum(lam1,A,w,N_sum)
	else:
		A = A - alpha*get_gradient_A_sum(lam1,A,m,w,N_sum)
	print(loss)


def prune(A,m,w,w_hat):
	return A,w_hat


M = round_sum(A,N_sum)
print("M is")
print(M)
M = np.round(M)
assert (M >= 0).all(), "Found negative entry in M matrix"
# Finally, use computed A and least squares to compute needed synaptic weights w = A*m
m = np.linalg.lstsq(M,w,rcond=None)[0]

if((m[0] > 0 and m[1] > 0) or (m[0] < 0 and m[1] < 0)):
	raise Exception("Error: Found two weights of the same nature.")

print("m is")
print(m)
print("Reconstructed weights")
w_hat = np.matmul(M,m)
print(w_hat)
print("True weights")
print(w)

A,w_hat = prune(M,m,w,w_hat)

