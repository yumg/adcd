package cn.neu.aiops.model.train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import cn.neu.aiops.model.conf.DynamicGCNAdjacentProvider;
import cn.neu.aiops.model.conf.DynamicGCNLayerConf;

public class TrainNGcn {

    private static RecordReaderDataSetIterator metricsBatchItr;
    private static List<INDArray> adjList;
    private static DataNormalization normalizer;
    private static GCNAdjacentProviderImpl adjacentProvider;
    private static INDArray adjacent;

//    final private static String modelFile = Util.WORK_DIR + "brain-split-NGcn.model";
//    private static String nodesMetricsFile = "brain-split-nodes-1.csv";
//    private static String adjecentsFile = "brain-split-adj-1.csv";
//    private static int batchSize = 7;

    final private static String modelFile = Util.WORK_DIR + "NGcn.model";
    private static String nodesMetricsFile = "NodesMetrics.csv";
    private static String adjecentsFile = "AdjacentSeq.csv";
    private static int batchSize = 5;

    static {
        CSVRecordReader metricsReader = new CSVRecordReader(',');
        try {
            metricsReader.initialize(new FileSplit(new File(Util.WORK_DIR + nodesMetricsFile)));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Schema inputDataSchema = new Schema.Builder().addColumnLong("DateTime")
                .addColumnsFloat("Metric1", "Metric2", "Metric3").addColumnInteger("Label").build();
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema).removeColumns("DateTime").build();
        int labelIndex = 3;
        int numPossibleLabels = 2;
        metricsBatchItr = new RecordReaderDataSetIterator(new TransformProcessRecordReader(metricsReader, tp),
                batchSize, labelIndex, numPossibleLabels);

        normalizer = new NormalizerStandardize();
        normalizer.fit(metricsBatchItr);

        CSVRecordReader adjacentsReader = new CSVRecordReader(',');
        try {
            adjacentsReader.initialize(new FileSplit(new File(Util.WORK_DIR + adjecentsFile)));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        adjList = new ArrayList<>();
        while (adjacentsReader.hasNext()) {
            byte[] bytes = Base64.getDecoder().decode(adjacentsReader.next().get(1).toString());
            INDArray indArray = Nd4j.fromByteArray(bytes);
            INDArray eye = Nd4j.eye(indArray.rows());
            indArray.addi(eye);
            indArray = Util.normalize(indArray);
            adjList.add(indArray);
        }
        try {
            adjacentsReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        adjacentProvider = new GCNAdjacentProviderImpl();
    }

    public static Iterator<Pair<DataSet, INDArray>> graphDataIterator() throws IOException, InterruptedException {
        metricsBatchItr.reset();
        Iterator<INDArray> adjacentItr = adjList.iterator();

        return new Iterator<Pair<DataSet, INDArray>>() {

            @Override
            public boolean hasNext() {
                return metricsBatchItr.hasNext();
            }

            @Override
            public Pair<DataSet, INDArray> next() {
                org.nd4j.linalg.dataset.DataSet dataSet = metricsBatchItr.next();
                INDArray adjacent = adjacentItr.next();
                Pair<DataSet, INDArray> rtv = new Pair<DataSet, INDArray>();
                normalizer.transform(dataSet);
                rtv.setFirst(dataSet);
                rtv.setSecond(adjacent);
                return rtv;
            }

        };
    }

    public static void main(String[] args) throws IOException, InterruptedException {
//        train();
        load();
    }

    static public void load() throws IOException, InterruptedException {
        MultiLayerNetwork net = MultiLayerNetwork.load(new File(modelFile), true);

        Iterator<Pair<DataSet, INDArray>> graphDataIterator = graphDataIterator();

        while (graphDataIterator.hasNext()) {
            Pair<DataSet, INDArray> next = graphDataIterator.next();
            adjacent = next.getSecond();
            DataSet m = next.getFirst();
            INDArray output = net.output(m.getFeatures());
            Evaluation eval = new Evaluation();
            eval.eval(output, m.getLabels());
            System.out.println(eval.stats());
        }
    }

    static public void train() throws IOException, InterruptedException {

        MultiLayerNetwork net = buildModel();

        int nEpochs = 2000; // number of training epochs
        for (int i = 0; i < nEpochs; i++) {
            Iterator<Pair<DataSet, INDArray>> graphDataIterator = graphDataIterator();
            while (graphDataIterator.hasNext()) {
                Pair<DataSet, INDArray> ds = graphDataIterator.next();
                adjacent = ds.getSecond();
                DataSet metrics = ds.getFirst();
                net.fit(metrics);
            }
        }

//
        Iterator<Pair<DataSet, INDArray>> graphDataIterator = graphDataIterator();
        Pair<DataSet, INDArray> next = graphDataIterator.next();
        adjacent = next.getSecond();
        DataSet m = next.getFirst();
        INDArray output = net.output(m.getFeatures());
        Evaluation eval = new Evaluation();
        eval.eval(output, m.getLabels());
        System.out.println(eval.stats());

        net.save(new File(modelFile));

//		int size = tmpLabels.size() / nEpochs;
//		INDArray labels = Nd4j.zeros(size * 5, 2);
//		for (int i = 0; i < size; i++) {
//			INDArray label = tmpLabels.get(i);
//			for (int j = 0; j < label.rows(); j++) {
//				labels.putSlice(i * 4 + j, label.getRow(j));
//			}
//		}
//
    }

    static public MultiLayerNetwork buildModel() throws IOException {
        int seed = 6; // number used to initialize a pseudorandom number generator.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).seed(seed)
//				.updater(new Adam(0.01)).seed(seed)// .dropOut(0.5)
                .l2(1e-4).weightInit(WeightInit.XAVIER).list()
                .layer(new DynamicGCNLayerConf.Builder().nIn(3).nOut(6).setAdjacentProvider(adjacentProvider)
                        .activation(Activation.RELU).build())
//                .layer(new DynamicGCNLayerConf.Builder().nIn(6).nOut(6).setAdjacentProvider(adjacentProvider)
//                        .activation(Activation.RELU).build())
                .layer(new DynamicGCNLayerConf.Builder().nIn(6).nOut(6).setAdjacentProvider(adjacentProvider)
                        .activation(Activation.SOFTMAX).build())
//				.layer(new LossLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(6).nOut(2).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
//
        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(new ScoreIterationListener(100));
//
        System.out.println(net.summary());
        return net;
    }

    static class GCNAdjacentProviderImpl implements DynamicGCNAdjacentProvider {
        /**
         *
         */
        private static final long serialVersionUID = 8919673050282079970L;

        @Override
        public INDArray getAdjacent() {
            return adjacent;
        }
    }
}
