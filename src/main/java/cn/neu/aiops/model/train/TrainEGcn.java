package cn.neu.aiops.model.train;

import java.io.File;
import java.io.IOException;

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

public class TrainEGcn {

	private static RecordReaderDataSetIterator metricsBatchItr;
	private static DataNormalization normalizer;
	private static GCNAdjacentProviderImpl adjacentProvider;
	private static INDArray adjacent;
	final private static String modelFile = Util.WORK_DIR + "EGcn.model";
	
	static {
		CSVRecordReader metricsReader = new CSVRecordReader(',');
		try {
			metricsReader.initialize(new FileSplit(new File(Util.WORK_DIR + "EdgesMetrics3.csv")));
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		Schema inputDataSchema = new Schema.Builder().addColumnLong("DateTime")
				.addColumnsFloat("Metric1", "Metric2", "Metric3").addColumnInteger("Label").build();
		TransformProcess tp = new TransformProcess.Builder(inputDataSchema).removeColumns("DateTime").build();
		int batchSize = 5;
		int labelIndex = 3;
		int numPossibleLabels = 2;
		metricsBatchItr = new RecordReaderDataSetIterator(new TransformProcessRecordReader(metricsReader, tp),
				batchSize, labelIndex, numPossibleLabels);

		normalizer = new NormalizerStandardize();
		normalizer.fit(metricsBatchItr);

		adjacentProvider = new GCNAdjacentProviderImpl();
		INDArray adjMatrix = Nd4j.readTxt(Util.WORK_DIR + "LineGraphAdjacent.csv");
		adjMatrix.addi(Nd4j.eye(adjMatrix.rows()));
		adjacent = Util.normalize(adjMatrix);
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		train();
//		load();
	}
	
	public static void load() throws IOException {
		MultiLayerNetwork net = MultiLayerNetwork.load(new File(modelFile), true);
		DataSet m = metricsBatchItr.next();
		normalizer.transform(m);
		INDArray output = net.output(m.getFeatures());
		Evaluation eval = new Evaluation();
		eval.eval(output, m.getLabels());
		System.out.println(eval.stats());
	}

	public static void train() throws IOException {

		MultiLayerNetwork net = buildModel();

		int nEpochs = 100; // number of training epochs
		for (int i = 0; i < nEpochs; i++) {
			while (metricsBatchItr.hasNext()) {
				DataSet metrics = metricsBatchItr.next();
				normalizer.transform(metrics);
				net.fit(metrics);
			}
			metricsBatchItr.reset();
		}

		DataSet m = metricsBatchItr.next();
		normalizer.transform(m);
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
				.layer(new DynamicGCNLayerConf.Builder().nIn(3).nOut(3).setAdjacentProvider(adjacentProvider)
						.activation(Activation.RELU).build())
				.layer(new DynamicGCNLayerConf.Builder().nIn(3).nOut(3).setAdjacentProvider(adjacentProvider)
						.activation(Activation.SOFTMAX).build())
//				.layer(new LossLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nIn(3).nOut(2).build())
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
		private static final long serialVersionUID = -6871424748837727817L;

		@Override
		public INDArray getAdjacent() {
			return adjacent;
		}

	}
}
