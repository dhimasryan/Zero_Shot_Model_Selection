import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
from keras import backend as K
from scipy.io import wavfile
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,  EarlyStopping, Callback
from keras import backend as K
from keras.constraints import unit_norm
from keras.layers import LSTM, TimeDistributed, Bidirectional
import scipy.io
import librosa
import argparse
import time, os, pickle
import numpy as np
import numpy.matlib
import math, random
import pdb
import sys

random.seed(999)
batch_size=1

def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    return FileList;
	
def make_spectrum_with_phase(y, Noisy=False):
    
    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    phase = np.angle(F)
    Lp = np.log10(np.abs(F)**2)
   
    if Noisy==True:
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase

def make_spectrum_for_dln(y):
    y=y/np.max(abs(y))
    F = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    Lp=np.abs(F)
    NLp=np.reshape(Lp.T,(1,Lp.shape[1],257))
    return NLp

def recons_spec_phase(mag,phase):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming)
    return result   

def convert_to_dln_input(x_enh):
    x_out=make_spectrum_for_dln(x_enh)
    
    return x_out
	
def testing(model,noisy_LP,noisy_phase,wave_name):	
    mask_LP=(model.predict(noisy_LP, verbose=0, batch_size=batch_size))
    enhanced_LP=np.squeeze(noisy_LP+mask_LP)
    enhanced_wav=recons_spec_phase(np.sqrt(10**(enhanced_LP.T)),noisy_phase)
    enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
    librosa.output.write_wav(os.path.join("CNN_enhanced_MSE",wave_name), enhanced_wav, 16000)

def Quality_Selection(Test_Noisy_paths, Test_Noisy_wavename):
    os.system("mkdir CNN_enhanced_MSE/") 
    print ('Quality-Net loading...')
	
    MdNamePath='Quality-Net_(Non-intrusive)'
    with open(MdNamePath+'.json', "r") as f:
         Quality = model_from_json(f.read());
	
    Quality.load_weights(MdNamePath+'.hdf5');
	
    print 'load SE model 1' 
    MdNamePath_1 = 'WSJ_CNN_2D_cluster_1_4class_50epoch_v2_irm'
    with open(MdNamePath_1+'.json', "r") as f:
        model_1 = model_from_json(f.read());	
    model_1.load_weights(MdNamePath_1+'.hdf5');
	
    print 'load SE model 2' 
    MdNamePath_2 = 'WSJ_CNN_2D_cluster_2_4class_50epoch_v2_irm'
    with open(MdNamePath_2+'.json', "r") as f:
        model_2 = model_from_json(f.read());	
    model_2.load_weights(MdNamePath_2+'.hdf5');
	
    print 'load SE model 3' 
    MdNamePath_3 = 'WSJ_CNN_2D_cluster_3_4class_50epoch_v2_irm'
    with open(MdNamePath_3+'.json', "r") as f:
        model_3 = model_from_json(f.read());	
    model_3.load_weights(MdNamePath_3+'.hdf5');
	
    print 'load SE model 4' 
    MdNamePath_4 = 'WSJ_CNN_2D_cluster_4_4class_50epoch_v2_irm'
    with open(MdNamePath_4+'.json', "r") as f:
        model_4 = model_from_json(f.read());	
    model_4.load_weights(MdNamePath_4+'.hdf5');


    for path in Test_Noisy_paths:   
        S=path.split('/')
        wave_name=S[-1]
		
        noisy, rate  = librosa.load(path,sr=16000)
        noisy_LP, noisy_phase = make_spectrum_with_phase(noisy,Noisy=True) 
        noisy_LP_DLN = convert_to_dln_input(noisy)
	
        [PESQ, frame_score]=Quality.predict(noisy_LP_DLN, verbose=0, batch_size=1)

        if PESQ > 3.375:
           se_model = model_1

        elif PESQ > 2.25 :
           se_model = model_2
		   
        elif PESQ > 1.25 :
           se_model = model_3		

        else :
           se_model = model_4
        
        testing(se_model,noisy_LP,noisy_phase,wave_name)
                  			
if __name__ == '__main__':	
    print 'testing ZMOS-QS...'
    
    Test_Noisy_paths = get_filenames('/data1/user_ryandhimas/MOSA-NET_Applications/List/' +sys.argv[1])
    Test_Noisy_wavename=[]
    
    for path in Test_Noisy_paths:
       S=path.split('/')[-1]
       Test_Noisy_wavename.append(S)
       
    Quality_Selection(Test_Noisy_paths, Test_Noisy_wavename)
    print 'complete testing stage'
