import os,sys
import numpy as np
import numpy.matlib
import pandas as pd
import librosa
import pickle
import joblib
import hdf5storage
from sklearn.cluster import KMeans

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    

def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    
    return FileList;
    
def testing_K_Means(hidden_list,noisy_list):	
    cluster_0_paths=[]
    cluster_1_paths=[]
    cluster_2_paths=[]
    cluster_3_paths=[]
	
    for index in range(len(hidden_list)):
      print index
      noisy_path = noisy_list[index]
	  
      hidden_output = hdf5storage.loadmat(hidden_list[index])
      hidden_output = hidden_output["hidden_output"]
      len_frame= len(hidden_output)
      
      average_value=0
      total_frame=0
      
      for i in range(len_frame):
          average_value+=hidden_output[i]
          total_frame+=1
      
      final_value=average_value/total_frame 
     
      df = pd.DataFrame(data=final_value)
      df = df.T

      filename = sys.argv[3]
      Kmean = joblib.load(filename)
      cluster=Kmean.predict(df)
      
      if cluster [0] == 0:
          cluster_0_paths.append(noisy_path)
      elif cluster [0] == 1:
          cluster_1_paths.append(noisy_path)
      elif cluster [0] == 2:
          cluster_2_paths.append(noisy_path)
      else:
          cluster_3_paths.append(noisy_path)

    with open(sys.argv[4], 'w') as f:
       for item in cluster_0_paths:
         f.write("%s\n" % item)

    with open(sys.argv[5], 'w') as g:
       for item in cluster_1_paths:
         g.write("%s\n" % item)

    with open(sys.argv[6], 'w') as f:
       for item in cluster_2_paths:
         f.write("%s\n" % item)

    with open(sys.argv[7], 'w') as g:
       for item in cluster_3_paths:
         g.write("%s\n" % item)

if __name__ == '__main__':   
    Hidden_output_paths = get_filenames("/Data/user_ryandhimas/Embedding_Q_NET/List/"+sys.argv[1])
    Noisy_paths = get_filenames("/Data/user_ryandhimas/Embedding_Q_NET/List/"+sys.argv[2])

    testing_K_Means(Hidden_output_paths,Noisy_paths)