# Introduction

This repository provides the partial source code and one dataset in our paper **"Anomaly detection for cloud systems with dynamic spatio-temporal learning"**.

## About the source code

For reasons of commercial confidentiality, we can’t publish the code of the whole system. 

We just can provide “partial source code”, which refers to the GCN layer implementation for Deeplearning4j.
Deeplearning4j is a suite of tools for deploying and training deep learning models using the JVM.
When we used it in our program, it did not have GCN layer yet.  Thus, we extended our implementation for it.

### Code Samples

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam(0.01)).seed(seed).dropOut(0.5)
        .l2(0.0005)
        .biasInit(0) 
        .miniBatch(false).list()
        .layer(new GCNLayerConf.Builder().nIn(32).nOut(128).adjacentMatrix(adjacent).activation(Activation.RELU)
                .weightInit(new UniformDistribution(0, 1)).build())
        .layer(new GCNLayerConf.Builder().nIn(128).nOut(64).adjacentMatrix(adjacent).activation(Activation.RELU)
                .weightInit(new UniformDistribution(0, 1)).build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nIn(64).nOut(2).build())
        .build();

MultiLayerNetwork net = new MultiLayerNetwork(conf);
net.init();
```

## About the dataset

Our paper used four datasets, one of which named DS-ESC is published in this repository.

DS-ESC is from the operational monitoring data of an elasticsearch cluster. 
The cluster consists of 7 nodes and provides data indexing services for software development. 
It indexes about 200-500 MB size of incremental data per day and has stored more than 200 indices with about 500 GB size of stock data.
We use the processes of elasticsearch as topological nodes and the communications among the processes as topological connections. 
We use the Cluster-API provided by elasticsearch to collect the node states, which include the metrics of the OS, JVM, and various pooled resources for the process. 
We use the iftop command to capture the communications between specific ports holded by the processes, then monitor the number of network links and the amount of data exchanged.

To increase data sampling efficiency, we randomly submitted some read and write requests that could easily overload the cluster. Those intentional requests, such as unreasonable parameters, actually sometimes can be received at the server side. That could easily cause the service runs out of order. At the same time, we tracked the service status of the cluster using dial testing. In this way, we got the anomalous states and used them as the labels of anomalies. 


