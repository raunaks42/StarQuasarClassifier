import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from mlxtend.evaluate import bias_variance_decomp as bvd
from concurrent.futures import ProcessPoolExecutor
import operator
import time
import os
import warnings
warnings.filterwarnings('ignore')

def loadDatset(filname, cVal=False):
        dset=pd.read_csv(filname)
        if cVal:
            y=dset['class']
            x=dset['spectrometric_redshift']
            x = x.to_numpy()
        else:
            if len(dset.columns)==38:
                dset=dset.drop(dset.columns[31:],axis=1)
                y=dset['class']
                dset=dset.drop(dset.columns[13:16],axis=1)
                x=dset.drop(dset.columns[0:7],axis=1)
            else:
                dset=dset.drop(dset.columns[30:],axis=1)
                y=dset['class']
                dset=dset.drop(dset.columns[13:15],axis=1)
                x=dset.drop(dset.columns[0:7],axis=1)
            sc = MinMaxScaler(feature_range=(0, 1))
            x = sc.fit_transform(x)

        y = y.to_numpy()
        return x,y

class mykNearstNeighs:
    def __init__(self, k):
        self.k=k

    def fit(self, traingSetx, traingSety):
        self.traingSetx=traingSetx
        self.traingSety=traingSety
        return self

    def predict(self, tstSetx):
        prdicns=[]
        for x in range(len(tstSetx)):
            neighbrs = self.getNeighs(self.traingSetx, self.traingSety, tstSetx[x], self.k)
            resp = self.getMajrtyVote(neighbrs)
            prdicns.append(resp)
        return prdicns

    def getNeighs(self, traingSetx, traingSety, tstInst, k):
        distncs = []
        for x in range(len(traingSetx)):
            distncs.append( ( traingSety[x] , self.manhatDistnc(tstInst,traingSetx[x]) ) )

        distncs.sort(key=operator.itemgetter(1))
        kneighs=[x[0] for x in distncs[:k]]

        return kneighs
    
    def manhatDistnc(self, instnc1, instnc2):
	    return np.sum(np.absolute(instnc1-instnc2))
    
    def getMajrtyVote(self, kneighs):
        clsVotes = [0,0]
        for x in range(len(kneighs)):
            resp = kneighs[x]
            clsVotes[resp] += 1
        return int(clsVotes[1]>clsVotes[0])

def kNearstNeighs(Setx, Sety, traingSetx, traingSety, k):
    mykNN=mykNearstNeighs(k)
    mykNN.fit(traingSetx,traingSety)
    prdicns=mykNN.predict(Setx)

    return f1_score(Sety.tolist(),prdicns,average='weighted')*100.0 , prdicns

def kNNwrap(filname):
    totSetx,totSety=loadDatset(filname)
    traingSetx,tstSetx,traingSety,tstSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
    '''print(filname,repr(len(tstSetx)))
    accurcy, prdictdMod = kNearstNeighs(tstSetx, tstSety, traingSetx, traingSety, 5)

    totSetx,totSety=loadDatset(filname,True)
    traingSetx,tstSetx,traingSety,tstSety=train_test_split(totSetx,totSety, test_size = 0.25, random_state = 69)
    prdictdRedShift=[]
    for i in tstSetx:
        prdictdRedShift.append((1 if i>=0.004 else 0))

    crossvalfscore=f1_score(prdictdMod,prdictdRedShift,average='weighted')*100.0'''
    print(filname+'\nAverage Expected Loss=%d; Average Bias=%d; Average Variance=%d\n' % bvd(mykNearstNeighs(1), traingSetx, traingSety, tstSetx, tstSety, num_rounds=75, random_seed=69), flush=True)
    print(filname+'\nAverage Expected Loss=%d; Average Bias=%d; Average Variance=%d\n' % bvd(mykNearstNeighs(5), traingSetx, traingSety, tstSetx, tstSety, num_rounds=75, random_seed=69), flush=True)
    print(filname+'\nAverage Expected Loss=%d; Average Bias=%d; Average Variance=%d\n' % bvd(mykNearstNeighs(15), traingSetx, traingSety, tstSetx, tstSety, num_rounds=75, random_seed=69), flush=True)

def main():
    dirctry=os.fsencode('.')
    initime=time.time()
    filnames=[]
    for F in os.listdir(dirctry):  #each file in directory
        filname=os.fsdecode(F)  #get filename
        if filname.endswith('.csv'):
            #kNNwrap(filname)
            filnames.append(filname)
    with ProcessPoolExecutor(8) as executr:
        executr.map(kNNwrap,filnames)

    print('Total time elapsed:', time.time()-initime)

main()

'''
k=5,cat3
b=0.036
v=0.011
'''
