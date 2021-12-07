"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os,sys
import random
import pdb
import pickle
import hdf5storage
from sklearn.cluster import KMeans
import sys
random.seed(999)

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    

def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    
    return FileList;
    
def train_data_generator(noisy_list):	
     
    for index in range(len(noisy_list)):
      print index
      hidden_output = hdf5storage.loadmat(noisy_list[index])
      hidden_output = hidden_output["hidden_output"]
      
      len_frame= len(hidden_output)
      
      average_value=0
      total_frame=0
      for i in range(len_frame):
          average_value+=hidden_output[i]
          total_frame+=1
      
      final_value=average_value/total_frame
    
      if index == 0:
        df = pd.DataFrame(data=final_value)
        df = df.T
          
      else:
        df2 = pd.DataFrame(data=final_value)
        df2 = df2.T
        df.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df = pd.concat([df,df2])

    return df

if __name__ == '__main__':   
    Train_Noisy_paths = get_filenames("/Lists/"+sys.argv[1])

    # data_shuffle
    Num_traindata=len(Train_Noisy_paths)
    permute = range(Num_traindata)
    random.shuffle(permute)
    Train_Noisy_paths=shuffle_list(Train_Noisy_paths,permute)


    utterances=train_data_generator(Train_Noisy_paths)
    Kmean = KMeans(n_clusters=4, random_state=0)
    print 'K-Means training ...'
    Kmean.fit(utterances)

    print 'Saving Model ...'
    filename = sys.argv[2]
    pickle.dump(Kmean, open(filename, 'wb'))

    Kmean.cluster_centers_
    Kmean.labels_