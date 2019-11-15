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


def prune(A,m,w,w_hat,N,fan_in):
	A = np.round(A)

	exc_syn_weight = max(m)
	inh_syn_weight = min(m)
	exc_syn_index = np.argmax(m)
	inh_syn_index = np.argmin(m)
	count = 0
	for i in range(A.shape[0]): #! Wrap evth in while changed smth loop
		# Check if we can remove an excitatory weight to improve the difference
		changed_smth = True
		if(i % N == 0):
			count = np.sum(np.abs(A[i:i+N,:])) # Count has to stay below 64
			if(count > 64):
				print("Error: More than fan-in number of weights.")
				changed_smth = False
			equal_share = int((fan_in-count) / N)
			
		per_neuron_cnt = 0
		while(changed_smth):
			changed_smth = False
			if(A[i,exc_syn_index] >0 and abs(w[i]-w_hat[i]) >= abs(w[i]-(w_hat[i]-exc_syn_weight))):
				#print("Removed excitatory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],exc_syn_weight))
				w_hat[i] -= exc_syn_weight
				A[i,exc_syn_index] -= 1
				changed_smth = True
				count -= 1
				per_neuron_cnt -= 1
			# Check if we can remove an inhibitory synaps
			elif(A[i,inh_syn_index] >0 and abs(w[i]-w_hat[i]) >= abs(w[i]-(w_hat[i] + abs(inh_syn_weight)))):
				#print("Removed inhibitory synapse. w: %.3f w_hat: %.3f Inh.syn.weight: %.3f" % (w[i],w_hat[i],inh_syn_weight))
				w_hat[i] += abs(inh_syn_weight)
				A[i,inh_syn_index] -= 1
				changed_smth = True
				count -= 1
				per_neuron_cnt -= 1
			# Check if we can add an excitatory weight
			elif(abs(w[i]-w_hat[i]) > abs(w[i] - (w_hat[i]+exc_syn_weight)) and count < fan_in and per_neuron_cnt < equal_share):
				#print("Added excitatory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],exc_syn_weight))
				w_hat[i] += exc_syn_weight
				A[i,exc_syn_index] += 1
				changed_smth = True
				count += 1
				per_neuron_cnt += 1
			# Check if we can add an inhibitory synapse
			elif(abs(w[i]-w_hat[i]) > abs(w[i] - (w_hat[i]+inh_syn_weight)) and count < fan_in and per_neuron_cnt < equal_share):
				#print("Added inhibitory synapse. w: %.3f w_hat: %.3f Exc.syn.weight: %.3f" % (w[i],w_hat[i],inh_syn_weight))
				w_hat[i] += inh_syn_weight
				A[i,inh_syn_index] += 1
				changed_smth = True
				count += 1
				per_neuron_cnt += 1
	
	return A,w_hat


#x = np.linspace(-3,3,100)
#y = x - np.sin(2*np.pi*x)/(2*np.pi)
#plt.plot(x,y)
#plt.show()

#np.random.seed(42)
np.set_printoptions(precision=5, suppress=True)

async def learn(N,N_sum,T,lam1,lam2,alpha,w,m,A,fan_in):
	for i in range(T):
		loss = get_loss(lam1,A,m,w)
		if(i % 2 == 0):
			m = m - alpha*get_gradient_m_sum(lam1,A,w,m,N_sum)
		else:
			A = A - alpha*get_gradient_A_sum(lam1,A,m,w,N_sum)
		#print("Scaled loss is %.6f" % (loss/N**2))

		
	M = round_sum(A,N_sum)
	M = np.round(M)

	if((m[0] >= 0 and m[1] <= 0) or (m[0] <= 0 and m[1] >= 0) and (M >= 0).all()):
		m = np.linalg.lstsq(M,w,rcond=None)[0]
		w_hat = np.matmul(M,m)
		M,w_hat = prune(M,m,w,w_hat,N,fan_in)
		final_error = np.linalg.norm(w-w_hat,2)/N**2
	else:
		final_error = np.inf
	print(final_error)

	return M,m,final_error

async def naive(N,w,fan_in):
	number_conns_per_neuron = int(fan_in / N)
	

async def main():

	N = 20
	N_sum = 100
	T = 500
	lam1 = 1
	lam2 = 1
	alpha = 0.001
	fan_in = 512
	w = np.random.randn(N**2).reshape((-1,1))
	m = np.random.randn(2).reshape((-1,1))
	A = np.abs(np.random.randn(N**2,2))

	num_tasks = 20
	tasks = []
	for t in range(num_tasks):
		#m = np.random.randn(2).reshape((-1,1))
		m = np.asarray([min(w),max(w)]).reshape((-1,1))
		A = np.abs(np.random.randn(N**2,2))
		tasks.append(asyncio.create_task(learn(N,N_sum,T,lam1,lam2,alpha,w,m,A,fan_in)))

	final_errors = np.zeros(num_tasks)
	Ms = []
	ms = []
	for idx,task in enumerate(tasks):
		M,m,loss = await task
		final_errors[idx] = loss
		Ms.append(M)
		ms.append(m)

	min_idx = np.argmin(final_errors)
	M = Ms[min_idx]
	m = ms[min_idx]
	print("Weights are"); print(m)
	
	M_res = np.sum(M, axis=1).reshape((N,N))
	print(M_res.astype(np.int))


	print(final_errors)
	w_hat = np.matmul(M,m)
	w = w.reshape((N,N))
	w_hat = w_hat.reshape((N,N))
	print("True weights: "); print(w)
	print("Reconstructed weights: "); print(w_hat)

	plt.figure(figsize=(5,10))
	plt.subplot(211)
	plt.matshow(w,fignum=False, vmin=np.min(w),vmax=np.max(w))
	plt.colorbar()
	plt.subplot(212)
	plt.matshow(w_hat,fignum=False, vmin=np.min(w),vmax=np.max(w))
	plt.colorbar()
	#plt.tight_layout()
	plt.show()

asyncio.run(main())