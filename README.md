# Logit Calibration for Non-IID and Long-Tailed Data in Federated Learning

This is the code for paper: **Logit Calibration for Non-IID and Long-Tailed Data in Federated Learning**

**Abstract:** Federated learning (FL) strives to enable collaborative training of deep models on the distributed clients of different data without centrally aggregating raw data and hence improving data privacy. Nevertheless, a central challenge in training classification models in the federated system is learning with non-IID data. Most of the existing work is dedicated to eliminating the heterogeneous influence of non-IID data in a federated system. However, in many real-world FL applications, the co-occurrence of data heterogeneity and long-tailed distribution is unavoidable. The universal class distribution is long-tailed, causing them to become easily biased towards head classes, which severely harms the global model performance. In this work, we also discovered an intriguing fact that the classifier logit vector (i.e., pre-softmax output) introduces a heterogeneity drift during the learning process of local training and global optimization, which harms the convergence as well as model performance. Therefore, motivated by the above finding, we propose a novel logit calibration FL method to solve the joint problem of non-IID and long-tailed data in federated learning, called Federated Learning with Logit Calibration (FedLC). First, we presented a method to mitigate the local update drift by calculating the Wasserstein distance among adjacent client logits and then aggregating similar clients to regulate local training. Second, based on the model ensemble, a new distillation method with logit calibration and class weighting was proposed by exploiting the diversity of local models trained on heterogeneous data, which effectively alleviates the global drift problem under long-tailed distribution. Finally, we evaluated FedLC using a highly non-IID and long-tailed experimental setting, comprehensive experiments on several benchmark datasets demonstrated that FedLC achieved superior performance compared with state-of-the-art FL methods, which fully illustrated the effectiveness of the logit calibration strategy. 



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- Mini-ImageNet



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `num_classes`               | Number of classes                                            |
| `num_clients`               | Number of all clients.                                       |
| `num_online_clients`        | Number of participating local clients.                       |
| `num_rounds`                | Number of communication rounds.                              |
| `num_data_train`            | Number of training data.                                     |
| `num_epochs_local_training` | Number of local epochs.                                      |
| `batch_size_local_training` | Batch size of local training.                                |
| `server steps`              | Number of steps of  training calibrated network.             |
| `distillation steps`        | Number of distillation steps.                                |
| `lr_global_teaching`        | Learning rate of server updating.                            |
| `lr_local_training`         | Learning rate of client updating.                            |
| `non_iid_alpha`             | Control the degree of non-IID.                               |
| `imb_factor`                | Control the degree of imbalance.                             |
| `ld`                        | Control the trade-off between $L_{CE}$ and $\lambda L_{KL}.$ |



### Usage

Here is an example to run FedLC on CIFAR-10 with imb_factor=0.01:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_data_training=49000 \
--num_epochs_local_training=10 \
--batch_size_local_training=128 \
--server_steps=100 \
--distillation_steps=100 \
--lr_global_training=0.001 \
--lr_local_training=0.1 \
--non-iid_alpha=0.1 \
--imb_factor=0.01 \ 
--ld=0.5
```



