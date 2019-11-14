import numpy as np
import matplotlib.pyplot as plt
import asyncio


def round_sum(x,N_sum):
	sum = 0
	for n in range(1,N_sum+1):
		sum += (-1)**n/(np.pi*n)*np.sin(2*n*np.pi*x)
	return x + sum

def rounding_func(A):
	return A - np.sin(2*np.pi*A)/(2*np.pi)

def get_gradient_A_sum(lam1,A,m,w,N_sum):
	t0 = np.zeros(A.shape)
	for n in range(1,N_sum+1):
		t0 += (-1)**n/(np.pi*n)*np.sin(2*np.pi*A)
	t5 = np.matmul(A + t0, m) - w
	t2 = np.zeros(A.shape)
	for n in range(1,N_sum+1):
		t2 += 2*(-1)**n*np.matmul(np.matmul(np.diagflat(t5),np.cos(2*n*np.pi*A)),np.diagflat(m))

	grad = np.matmul(t5,m.T) + t2
	return grad

def get_gradient_m_sum(lam1,A,w,m,N_sum):
	T0 = np.zeros(A.shape)
	for n in range(1,N_sum):
		T0 += (-1)**n/(np.pi*n)*np.sin(1*np.pi*n*A)

	T1 = A + T0
	grad = np.matmul(T1.T, np.matmul(T1,m)-w)
	return grad

def get_loss(lam1,A,m,w):
	return lam1/2*np.linalg.norm(np.matmul(round_sum(A,100),m)-w,2)**2


def prune(A,m,w,w_hat):
	A = np.round(A)

	exc_syn_weight = max(m)
	inh_syn_weight = min(m)
	exc_syn_index = np.argmax(m)
	inh_syn_index = np.argmin(m)

	for i in range(A.shape[0]): #! Wrap evth in while changed smth loop
		# Check if we can remove an excitatory weight to improve the difference
		changed_smth = True
		while(changed_smth):
			changed_smth = False
			if(A[i,exc_syn_index] >0 and abs(w[i]-w_hat[i]) >= abs(w[i]-(w_hat[i]-exc_syn_weight))):
				print("Removed excitatory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],exc_syn_weight))
				w_hat[i] -= exc_syn_weight
				A[i,exc_syn_index] -= 1
				changed_smth = True
			# Check if we can remove an inhibitory synaps
			elif(A[i,inh_syn_index] >0 and abs(w[i]-w_hat[i]) >= abs(w[i]-(w_hat[i] + abs(inh_syn_weight)))):
				print("Removed inhibitory synapse. w: %.3f w_hat: %.3f Inh.syn.weight: %.3f" % (w[i],w_hat[i],inh_syn_weight))
				w_hat[i] += abs(inh_syn_weight)
				A[i,inh_syn_index] -= 1
				changed_smth = True
			# Check if we can add an excitatory weight
			elif(abs(w[i]-w_hat[i]) > abs(w[i] - (w_hat[i]+exc_syn_weight))):
				print("Added excitatory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],exc_syn_weight))
				w_hat[i] += exc_syn_weight
				A[i,exc_syn_index] += 1
				changed_smth = True
			# Check if we can add an inhibitory synapse
			elif(abs(w[i]-w_hat[i]) > abs(w[i] - (w_hat[i]+inh_syn_weight))):
				print("Added inhibitory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],inh_syn_weight))
				w_hat[i] += inh_syn_weight
				A[i,inh_syn_index] += 1
				changed_smth = True
	
	return A,w_hat


#x = np.linspace(-3,3,100)
#y = x - np.sin(2*np.pi*x)/(2*np.pi)
#plt.plot(x,y)
#plt.show()

#np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)

def learn(N,N_sum,T,lam1,lam2,alpha,w,m,A):
	for i in range(T):
		loss = get_loss(lam1,A,m,w)
		if(i % 2 == 0):
			m = m - alpha*get_gradient_m_sum(lam1,A,w,m,N_sum)
		else:
			A = A - alpha*get_gradient_A_sum(lam1,A,m,w,N_sum)
		print("Scaled loss is %.4f" % (loss/N**2))

	if((m[0] > 0 and m[1] > 0) or (m[0] < 0 and m[1] < 0)):
		return_loss = np.inf
	else:
		return_loss = loss/N**2

	return A,m,return_loss

N = 2
N_sum = 100
T = 1000
lam1 = 1
lam2 = 1
alpha = 0.001
w = np.random.randn(N**2).reshape((-1,1))
m = np.random.randn(2).reshape((-1,1))
A = np.abs(np.random.randn(N**2,2))

A,_,loss = learn(N,N_sum,T,lam1,lam2,alpha,w,m,A)

M = round_sum(A,N_sum)

print("M is")
print(M)
M = np.round(M)

assert loss < np.inf, "Have not found a valid pair of weights"
assert (M >= 0).all(), "Found negative entry in M matrix"

print("Final loss is %.4f" % loss)

# Finally, use computed A and least squares to compute needed synaptic weights w = A*m
m = np.linalg.lstsq(M,w,rcond=None)[0]


print("m is")
print(m)
print("Reconstructed weights")
w_hat = np.matmul(M,m)
print(w_hat)
print("True weights")
print(w)

A,w_hat = prune(M,m,w,w_hat)

print("After pruning:")
print(A)

print("Reconstructed weights")
print(w_hat)

print("True weights")
print(w)

final_error = np.linalg.norm(w-w_hat,2)/N**2
print("Final error is %.5f" % final_error)