"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
import keras
from keras.models import Sequential, model_from_json, Model
import hdf5storage
import scipy.io
import scipy.stats
import librosa
import numpy as np
import numpy.matlib
import random
import pdb
random.seed(999)
batch_size=1

def feature_extraction(path, Noisy=False):
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal))
    
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

if __name__ == '__main__':   
    Noisy_list=ListRead('NoisyList.txt')
    random.shuffle(Noisy_list)
    Data_list= Noisy_list

    MdNamePath='Quality-Net_(Non-intrusive)'
    with open(MdNamePath+'.json', "r") as f:
        model = model_from_json(f.read());
	
    model.load_weights(MdNamePath+'.hdf5');
    pop_layer(model)
    pop_layer(model)

    index = 0
    for path in Data_list:   
        S=path.split('/')
        dB=S[-5]
        wave_name=S[-1]
        name = wave_name[:-4]
        #pdb.set_trace()
        noisy_LP, _ =feature_extraction(path)           
    
        Hidden_output=np.squeeze(model.predict(noisy_LP, verbose=0, batch_size=batch_size))
        #pdb.set_trace()
        hdf5storage.savemat('/Dir_/Hidden_Output_Quality_Net/'+name+'.mat', {'hidden_output':Hidden_output}, format='7.3');