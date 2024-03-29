# Speech Enhancement with Zero-Shot Model Selection

### Introduction ###

Recent research on speech enhancement (SE) has seen the emergence of deep-learning-based methods. It is still a challenging task to determine the effective ways to increase the generalizability of SE under diverse test conditions. In this study, we combine zero-shot learning and ensemble learning to propose a zero-shot model selection (ZMOS) approach to increase the generalization of SE performance. The proposed approach is realized in the offline and online phases. The offline phase clusters the entire set of training data into multiple subsets and trains a specialized SE model (termed component SE model) with each subset. The online phase selects the most suitable component SE model to perform the enhancement. Furthermore, two selection strategies were developed: selection based on the quality score (QS) and selection based on the quality embedding (QE). Both QS and QE were obtained using a Quality-Net, a non-intrusive quality assessment network. Experimental results confirmed that the proposed ZMOS approach can achieve better performance in both seen and unseen noise types compared to the baseline systems and other model selection systems, which indicates the effectiveness of the proposed approach in providing robust SE performance.

For more detail please check our <a href="https://arxiv.org/ftp/arxiv/papers/2012/2012.09359.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```
### Steps ###
1. Please follow each steps to run the code. 
2. The first stage is extracting embedding features (01_Extract_Latent_Representation.py), and followed by the K-Means training (02_K_Mean_Clustering_Training.py) and testing (03_K_Mean_Clustering_Testing.py) stage.
3. For developing the Quality-Net model, please kindly refer to <a href="https://github.com/JasonSWFu/Quality-Net" target="Quality-Net">Quality-Net</a>
4. For the speech enhancement model, you can use any model architecture. In our implementation, we used masking based CNN as our model architecture.
5. To get ZMOS-QS results, please use 04_ZMOS_QS.py
6. To get ZMOS-QE results, please use 05_ZMOS_QE.py

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, C.-S. Fuh, H.-M. Wang, and Y. Tsao “Speech Enhancement with Zero-Shot Model Selection,” to appear in EUSIPCO 2021
