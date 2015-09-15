from numpy import *
def loaddataset():
	datamat=[];labelmat=[]
	with open('testSet.txt') as fr:
		for line in fr.readlines():
			linearr=line.strip().split()
			datamat.append([1.0,float(linearr[0]),float(linearr[1])])
			labelmat.append(int(linearr[2]))
	return datamat,labelmat
	
def sigmod(inx):
	return 1.0/(1+exp(-inx))

def gradascent(datamatin,classlabels):
	datamat=mat(datamatin)
	labelmat=mat(classlabels).transpose()
	m,n=shape(datamat)
	alpha=0.01
	maxcycles=500
	weights=ones((n,1))
	for k in range(maxcycles):
		h=sigmod(datamat*weights)
		error=labelmat-h
		weights=weights+alpha*datamat.transpose()*error
	return weights

def plotbestfir(wei):
	import matplotlib.pyplot as plt
	weights=wei.getA()
	datamat,labelmat=loaddataset()
	dataarr=array(datamat)
	n=shape(dataarr)[0]
	xcord1=[];ycord1=[]
	xcord2=[];ycord2=[]
	for i in range(n):
		if int(labelmat[i])==1:
			xcord1.append(dataarr[i,1]);ycord1.append(dataarr[i,2])
		else:
			xcord2.append(dataarr[i,1]);ycord2.append(dataarr[i,2])
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x=arange(-3.0,3.0,0.1)
	y=(-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('xa'),plt.xlabel('X2')
	plt.show()

def randomgradascent0(datamat,classlabels):
	m,n=shape(datamat)
	alpha=0.01
	weights=ones(n)
	for i in range(m):
		h=sigmod(sum(datamat[i]*weights))
		error=classlabels[i]-h
		weights=weights+alpha*error*datamat[i]
	return weights

def randomgradascent1(datamat,classlabels,numiter=200):
	m,n=shape(datamat)
	weights=ones(n)
	for j in range(numiter):
		dataindex=range(m)
		for i in range(m):
			alpha=4/(1.0+j+i)+0.01
			randindex=int(random.uniform(0,len(dataindex)))
			h=sigmod(sum(datamat[randindex]*weights))
			print randindex
			error=classlabels[randindex]-h
			weights=weights+alpha*error*datamat[randindex]
			del(dataindex[randindex])
	return weights
