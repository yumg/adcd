# An implementation of the GCN layer to Deeplearning4j

Code Samples:

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