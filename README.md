## ElasticNetADMM

This is the source code for our paper: **面向物联网隐私数据分析的分布式弹性网络回归学习算法**. A brief introduction of this work is as follows:

>  In order to solve the problems caused by the traditional data analysis based on the centralized algorithm in the IoT, such as excessive bandwidth occupation, high communication latency and data privacy leakage, considering the typical linear regression model of elastic net regression, a distributed learning algorithm for Internet of Things (IoT) is proposed in this paper. This algorithm is based on the the Alternating Direction Method of Multipliers (ADMM) framework. It decomposes the objective problem of elastic net regression into several sub-problems that can be solved independently by each IoT node using its local data. Different from traditional centralized algorithms, the proposed algorithm does not require the IoT node to upload its private data to the server for training, but rather the locally trained intermediate parameters to the server for aggregation. In such a collaborative manner, the server can finally obtain the objective model after several iterations. The experimental results on two typical datasets indicate that the proposed algorithm can quickly converge to the optimal solution within dozens of iterations. As compared to the localized algorithm in which each node trains the model solely based on its own local data, the proposed algorithm improves the validity and the accuracy of training models; as compared to the centralized algorithm, the proposed algorithm can guarantee the accuracy and the scalability of model training, and well protect the individual private data from leakage.

中文摘要：

>  为了解决基于集中式算法的传统物联网数据分析处理方式易引发网络带宽压力过大、延迟过高以及数据隐私安全等问题，该文针对弹性网络回归这一典型的线性回归模型，提出一种面向物联网(IoT)的分布式学习算法。该算法基于交替方向乘子法(ADMM)，将弹性网络回归目标优化问题分解为多个能够由物联网节点利用本地数据进行独立求解的子问题。不同于传统的集中式算法，该算法并不要求物联网节点将隐私数据上传至服务器进行训练，而仅仅传递本地训练的中间参数，再由服务器进行简单整合，以这样的协作方式经过多轮迭代获得最终结果。基于两个典型数据集的实验结果表明：该算法能够在几十轮迭代内快速收敛到最优解。相比于由单个节点独立训练模型的本地化算法，该算法提高了模型结果的有效性和准确性；相比于集中式算法，该算法在确保计算准确性和可扩展性的同时，可有效地保护个体隐私数据的安全性。

This paper has been accepted and will be published by the Chinese journal 电子与信息学报, and can be downloaded from [here](http://jeit.ie.ac.cn/article/doi/10.11999/JEIT190739).

## Required software

Matlab

## Dataset

Please check the datasets in the ElasticNetADMM\Source Code\数据集 folder.

## Citation

Please visit the journal [webpage](http://jeit.ie.ac.cn/article/doi/10.11999/JEIT190739) for the citation information.

## Contact

Mengran Liu (18120381@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.

