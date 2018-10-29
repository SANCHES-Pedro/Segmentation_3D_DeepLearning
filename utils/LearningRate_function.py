from matplotlib.pyplot import *


#Starts with a cyclic learning rate and then an exponential decay with gamma
def lr_scheduler(epoch):
	lr_max = 0.0015
	lr_min = 0.0001
	step_size = 30
	gamma = 0.999
	change1 = 200
	change2 = 31*step_size
	if epoch<=change1:
		lr = lr_max
	elif change1<epoch<change2:
		lr = lr_cycle(epoch,lr_max,lr_min,step_size)
	else:
		lr = lr_exp(epoch,gamma,lr_min,change2)	
	print 'Learning rate: ',lr
	return lr

def lr_exp(epoch,gamma,init,change):
	return init*(gamma**(epoch-change))
	
#setting cyclic learning rate as a triangular function
def lr_cycle(epoch,lr_max= 0.001,lr_min = 0.0001,step_size= 15):
	step = epoch%(2*step_size)
	if step<step_size:
		lr = lr_max - step*(lr_max-lr_min)/step_size
	else:
		lr = lr_min + (step-step_size)*(lr_max-lr_min)/step_size
	return lr

def lr_finder(epoch):
	lr1 = 1e-8*(0.6**(-epoch)) #30 epochs
	print lr1
	return lr1
	
num = 30
x = range(num)
y = range(num)

for i in range(num):
	y[i]= lr_finder(i)
	x[i]=i

plot(x,y)
savefig('learning_rate.pdf')
show()
