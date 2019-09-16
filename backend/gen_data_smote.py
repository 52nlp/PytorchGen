# -*- coding: utf-8 -*-
__author__ = "Yu-Hsuan Chen (Albert)"
import datetime
import csv
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

data_set = "fraud"
data_dic = {"fraud": "./original_data/creditcard_1_train_no_label.csv"}

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            if self.newindex != 492:
                self.newindex += 1
            elif (self.newindex == 492):
                self.newindex = 0


def gen_fake_data_smote(n=30000, N=100, k=5):
    data = pd.read_csv(data_dic[data_set])
    now = datetime.datetime.now()
    a = data.values
    df_fake_data_combine = pd.DataFrame(columns=data.columns)
    # over_samples = N_100_K_5.over_sampling()
    #df_fake_data = pd.DataFrame(over_samples, columns=data.columns)
    for i in range((n // 492) + 1):
        N_100_K_5 = Smote(a, N=N, k=k)
        over_samples = N_100_K_5.over_sampling().copy()
        df_fake_data = pd.DataFrame(over_samples, columns=data.columns)
        # print(df_fake_data.head(3))
        df_fake_data_combine = df_fake_data_combine.append(df_fake_data, ignore_index=True)
    df_fake_data_combine = df_fake_data_combine.sample(n=n, random_state=12)
    df_fake_data_combine['Class'] = 1
    name = f"smote_{now.strftime('%Y-%m-%d')}_{N}_{k}_{str(n)}"
    df_fake_data_combine.to_csv(f"fake_data/{data_set}/{name}.csv", index=False)
    print("Output file:", f"fake_data/{data_set}/{name}.csv")
    return f"fake_data/{data_set}/{name}.csv"


#gen_fake_data = gen_fake_data_smote()
