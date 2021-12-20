import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.activations import softmax
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
import hdf5storage
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import pdb
import sys
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import joblib
random.seed(999)

batch_size=1
forgetgate_bias=-3

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
    
def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));

    return FileList;
  
def Sp_and_phase_Quality_Net(path, Noisy=False):
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

def make_spectrum_with_phase(y, Noisy=False):
    
    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    phase=np.angle(F)
    
    Lp = np.log10(np.abs(F)**2)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase
		
def recons_spec_phase(mag,phase):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming)
    return result            


def Quality_Embedding(Test_Noisy_paths, Test_Noisy_wavename):

    ######################## Model #################################
    MdNamePath='Quality-Net_(Non-intrusive)'
    with open(MdNamePath+'.json', "r") as f:
       model = model_from_json(f.read());
    model.load_weights(MdNamePath+'.hdf5');

    pop_layer(model)
    pop_layer(model)

    MdNamePath_SE0='WSJ_CNN_2D_cluster_1_new_WSJ_irm_100'
    with open(MdNamePath_SE0+'.json', "r") as f:
        se_model_0 = model_from_json(f.read());
    se_model_0.load_weights(MdNamePath_SE0+'.hdf5');

    MdNamePath_SE1='WSJ_CNN_2D_cluster_2_new_WSJ_irm_100'
    with open(MdNamePath_SE1+'.json', "r") as f:
        se_model_1 = model_from_json(f.read());
    se_model_1.load_weights(MdNamePath_SE1+'.hdf5');

    MdNamePath_SE2='WSJ_CNN_2D_cluster_3_new_WSJ_irm_100'
    with open(MdNamePath_SE2+'.json', "r") as f:
        se_model_2 = model_from_json(f.read());
    se_model_2.load_weights(MdNamePath_SE2+'.hdf5');

    MdNamePath_SE3='WSJ_CNN_2D_cluster_4_new_WSJ_irm_100'
    with open(MdNamePath_SE3+'.json', "r") as f:
        se_model_3 = model_from_json(f.read());
    se_model_3.load_weights(MdNamePath_SE3+'.hdf5');

    os.system("mkdir CNN_enhanced_MSE")
    
    for path in Test_Noisy_paths:   
        S=path.split('/')
        dB=S[-5]
        wave_name=S[-1]
        name = wave_name[:-4]

        noisy, rate  = librosa.load(path,sr=16000)
        noisy_LP, noisy_phase=make_spectrum_with_phase(noisy, Noisy=True)
        noisy_LP_Q_NET, _ =Sp_and_phase_Quality_Net(path)           
    
        hidden_output=np.squeeze(model.predict(noisy_LP_Q_NET, verbose=0, batch_size=batch_size))
        len_frame= len(hidden_output)
      
        average_value=0
        total_frame=0
    
        for i in range(len_frame):
            average_value+=hidden_output[i]
            total_frame+=1
      
        final_value=average_value/total_frame 
     
        df = pd.DataFrame(data=final_value)
        df = df.T

        filename = 'k_means_4class.sav'
        Kmean = joblib.load(filename)

        cluster=Kmean.predict(df)

        if cluster [0] == 0:
           mask_LP=se_model_0.predict(noisy_LP, verbose=0, batch_size=batch_size)
       
        elif cluster [0] == 1:
           mask_LP=se_model_1.predict(noisy_LP, verbose=0, batch_size=batch_size)
       
        elif cluster [0] == 2:
           mask_LP=se_model_2.predict(noisy_LP, verbose=0, batch_size=batch_size)
       
        else:
           mask_LP=se_model_3.predict(noisy_LP, verbose=0, batch_size=batch_size)
    
        enhanced_LP=np.squeeze(noisy_LP+mask_LP)	  
        enhanced_wav=recons_spec_phase(np.sqrt(10**(enhanced_LP.T)),noisy_phase)
        enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
        librosa.output.write_wav(os.path.join("CNN_enhanced_MSE",wave_name), enhanced_wav, 16000)
            
            
if __name__ == '__main__':	
    print 'testing ZMOS-QE'
    
    Test_Noisy_paths = get_filenames("/data1/user_ryandhimas/MOSA-NET_Applications/List/"+sys.argv[1])
    Test_Noisy_wavename=[]
    for path in Test_Noisy_paths:
       S=path.split('/')[-1]
       Test_Noisy_wavename.append(S)
       
    Quality_Embedding(Test_Noisy_paths, Test_Noisy_wavename)
    print 'complete testing stage'

