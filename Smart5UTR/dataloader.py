import pandas as pd
import numpy as np
import scipy.stats as stats

inp_len = 50
nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}

class RawData():
    def __init__(self, fname):
        self.df = pd.read_csv(fname)
        self.df = self.df.sort_values(by=['total'], ascending=False).reset_index(drop=True)
        self.df = self.df.loc[:220000 - 1, ['utr', 'total', 'rl']]

    def get_df(self):
        return self.df

    def get_seqs(self):
        return self.df['utr']

    def get_labels(self):
        return self.df['rl']

    def get_onehotmtxs(self):
        self.onehotmtx = onehot_coder(self.get_seqs())
        return self.onehotmtx

def onehot_coder(seqs):

    onehotmtx = np.empty([len(seqs), inp_len, 4])  ## init
    for i in range(len(seqs)):
        seq = seqs.iloc[i]
        seq = seq.lower()
        for n, x in enumerate(seq):
            onehotmtx[i][n] = np.array(nuc_d[x])
    return onehotmtx

def onehot_singleseq(seq):   ## input: single str, output: mtx with shape(1,50,4)
    onehotmtx = np.zeros([1, inp_len, 4])  ## init
    seq = seq.lower()
    for n, x in enumerate(seq):
        onehotmtx[0][n] = np.array(nuc_d[x])
    return onehotmtx

def binary_mtx(mtxs, seqlen=50):
    b_mtx = np.zeros([len(mtxs), seqlen, 4])
    for i in range(len(mtxs)):   ## mtx shape (50,4)
        for k in range(seqlen):
            base = mtxs[i][k].tolist()
            b_mtx[i][k][base.index(max(base))] = 1
    return b_mtx

def decode_seq(mtx, seqlen=50):  ## single matrix as input
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1] }
    seq=[]
    for i in range(seqlen):
        for x in ['a', 'c', 'g', 't']:
            if((mtx[i]==nuc_d[x]).all()):
                seq.append(x)
                break

    return "".join(seq)

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2