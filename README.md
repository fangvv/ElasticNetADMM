## ElasticNetADMM

This is the source code for our paper: **面向物联网隐私数据分析的分布式弹性网络回归学习算法**. A brief introduction of this work is as follows:

>  In order to solve the problems caused by the traditional data analysis based on the centralized algorithm in the IoT, such as excessive bandwidth occupation, high communication latency and data privacy leakage, considering the typical linear regression model of elastic net regression, a distributed learning algorithm for Internet of Things (IoT) is proposed in this paper. This algorithm is based on the the Alternating Direction Method of Multipliers (ADMM) framework. It decomposes the objective problem of elastic net regression into several sub-problems that can be solved independently by each IoT node using its local data. Different from traditional centralized algorithms, the proposed algorithm does not require the IoT node to upload its private data to the server for training, but rather the locally trained intermediate parameters to the server for aggregation. In such a collaborative manner, the server can finally obtain the objective model after several iterations. The experimental results on two typical datasets indicate that the proposed algorithm can quickly converge to the optimal solution within dozens of iterations. As compared to the localized algorithm in which each node trains the model solely based on its own local data, the proposed algorithm improves the validity and the accuracy of training models; as compared to the centralized algorithm, the proposed algorithm can guarantee the accuracy and the scalability of model training, and well protect the individual private data from leakage.

TBD

## Required software

Matlab

## Dataset

In our paper, we use the two well-known datasets as follows:

1. https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_example.html
2. https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

## Citation

TBD

## Contact

Mengran Liu (18120381@bjtu.edu.cn)

Weiwei Fang (fangvv@qq.com)
