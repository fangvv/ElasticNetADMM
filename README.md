## ElasticNetADMM

This is the source code for our paper: **面向物联网隐私数据分析的分布式弹性网络回归学习算法**. A brief introduction of this work is as follows:

>  In order to solve the problems caused by the traditional data analysis based on the centralized algorithm in the IoT, such as excessive bandwidth occupation, high communication latency and data privacy leakage, considering the typical linear regression model of elastic net regression, a distributed learning algorithm for Internet of Things (IoT) is proposed in this paper. This algorithm is based on the the Alternating Direction Method of Multipliers (ADMM) framework. It decomposes the objective problem of elastic net regression into several sub-problems that can be solved independently by each IoT node using its local data. Different from traditional centralized algorithms, the proposed algorithm does not require the IoT node to upload its private data to the server for training, but rather the locally trained intermediate parameters to the server for aggregation. In such a collaborative manner, the server can finally obtain the objective model after several iterations. The experimental results on two typical datasets indicate that the proposed algorithm can quickly converge to the optimal solution within dozens of iterations. As compared to the localized algorithm in which each node trains the model solely based on its own local data, the proposed algorithm improves the validity and the accuracy of training models; as compared to the centralized algorithm, the proposed algorithm can guarantee the accuracy and the scalability of model training, and well protect the individual private data from leakage.

This paper has been accepted and will be published by the Chinese journal 电子与信息学报, and can be downloaded from [here](http://jeit.ie.ac.cn/article/doi/10.11999/JEIT190739).

## Required software

Matlab

## Dataset

Please check the datasets in the ElasticNetADMM\Source Code\数据集 folder.

## Citation

Please visit the journal [webpage](http://jeit.ie.ac.cn/article/doi/10.11999/JEIT190739) for the citation information.

## Contact

Mengran Liu (18120381@bjtu.edu.cn)

Weiwei Fang (fangvv@qq.com)

