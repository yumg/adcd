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
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import cn.neu.aiops.model.conf.DynamicGCNLayerConf;
import cn.neu.aiops.model.conf.GCNLayerConf;

public class TrainNGcn2 {

	public static float[][] adjacent() throws IOException {
//		INDArray adjMatrix = Nd4j.readTxt(Util.WORK_DIR + "Adjacent.csv");
//		return adjMatrix.toFloatMatrix();
//
//		INDArray ones = Nd4j.ones(5, 5);
//		return ones.toFloatMatrix();

		float[][] adjMatrix_raw = Nd4j.zeros(5, 5).toFloatMatrix();
		adjMatrix_raw[0][1] = 1f;
		adjMatrix_raw[0][4] = 1f;
		adjMatrix_raw[1][0] = 1f;
		adjMatrix_raw[1][2] = 1f;
		adjMatrix_raw[2][1] = 1f;
		adjMatrix_raw[2][3] = 1f;
		adjMatrix_raw[3][2] = 1f;
		adjMatrix_raw[3][4] = 1f;
		adjMatrix_raw[4][3] = 1f;
		adjMatrix_raw[4][0] = 1f;
		return adjMatrix_raw;
	}

	public static DataSetIterator allData() throws IOException, InterruptedException {
		CSVRecordReader csvRecordReader = new CSVRecordReader(',');
		csvRecordReader.initialize(new FileSplit(new File(Util.WORK_DIR + "NodesMetrics3.csv")));
		Schema inputDataSchema = new Schema.Builder().addColumnLong("DateTime")
				.addColumnsFloat("Metric1", "Metric2", "Metric3").addColumnInteger("Label").build();
		TransformProcess tp = new TransformProcess.Builder(inputDataSchema).removeColumns("DateTime").build();
		return new RecordReaderDataSetIterator(new TransformProcessRecordReader(csvRecordReader, tp), 5, 3, 2);
	}

	public static List<INDArray> adjacents() throws IOException, InterruptedException {
		CSVRecordReader csvRecordReader = new CSVRecordReader(',');
		csvRecordReader.initialize(new FileSplit(new File(Util.WORK_DIR + "AdjacentSeq3.csv")));
		List<INDArray> rtv = new ArrayList<>();
		while (csvRecordReader.hasNext()) {
			byte[] bytes = Base64.getDecoder().decode(csvRecordReader.next().get(1).toString());
			INDArray indArray = Nd4j.fromByteArray(bytes);
			INDArray eye = Nd4j.eye(indArray.rows());
			indArray.addi(eye);
			indArray = Util.normalize(indArray);
			rtv.add(indArray);
		}
		csvRecordReader.close();
		return rtv;
	}

	public static void main(String[] args) throws IOException, InterruptedException {

		DataSetIterator allData = allData();

		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(allData); // Collect the statistics (mean/stdev) from the training data. This does not
									// modify the input data

		MultiLayerNetwork net = buildModel000();

		List<INDArray> adjacents = adjacents();
		Iterator<INDArray> adjItr = adjacents.iterator();

		List<INDArray> tmpLabels = new ArrayList<INDArray>();
		int nEpochs = 50; // number of training epochs
		for (int i = 0; i < nEpochs; i++) {
			while (allData.hasNext()) {
				INDArray next = adjItr.next();
//				DynamicGCNLayer.adjMatrix_ndarray = next;
				System.out.println(next);
				DataSet miniBatch = allData.next();
				tmpLabels.add(miniBatch.getLabels());
				normalizer.transform(miniBatch);
				System.out.println(miniBatch);
				net.fit(miniBatch);
			}
			allData.reset();
			adjItr = adjacents.iterator();
		}

		adjItr = adjacents.iterator();
//		DynamicGCNLayer.adjMatrix_ndarray = adjItr.next();

		allData.reset();
		DataSet next = allData.next();
		normalizer.transform(next);
		INDArray output = net.output(next.getFeatures());
		Evaluation eval = new Evaluation();
		eval.eval(output, next.getLabels());
		System.out.println(eval.stats());

//		int size = tmpLabels.size() / nEpochs;
//		INDArray labels = Nd4j.zeros(size * 5, 2);
//		for (int i = 0; i < size; i++) {
//			INDArray label = tmpLabels.get(i);
//			for (int j = 0; j < label.rows(); j++) {
//				labels.putSlice(i * 4 + j, label.getRow(j));
//			}
//		}
//
//		allData.reset();
//		INDArray output = net.output(allData);
//		Evaluation eval = new Evaluation();
//		eval.eval(output, labels);
//		System.out.println(eval.stats());

	}

	static public MultiLayerNetwork buildModel0() throws IOException {
		float[][] adjacent = adjacent();
		int seed = 6; // number used to initialize a pseudorandom number generator.
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.001)).seed(seed)
//				.updater(new Adam(0.01)).seed(seed)// .dropOut(0.5)
				.l2(1e-4).weightInit(WeightInit.XAVIER).list()
				.layer(new GCNLayerConf.Builder().nIn(3).nOut(3).adjacentMatrix(adjacent).activation(Activation.RELU)
						.build())

				.layer(new GCNLayerConf.Builder().nIn(3).nOut(3).adjacentMatrix(adjacent).activation(Activation.SOFTMAX)
						.build())

//				.layer(new LossLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())

				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nIn(3).nOut(2).build())

				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
//
		net.setListeners(new ScoreIterationListener(1));
//
		System.out.println(net.summary());
		return net;
	}

	public static MultiLayerNetwork buildModel1() {
		final int numInputs = 3;
		int outputNum = 2;
		long seed = 6;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()
				.layer(new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
				.layer(new DenseLayer.Builder().nIn(3).nOut(3).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX) // Override the global TANH activation with softmax for this
														// layer
						.nIn(3).nOut(outputNum).build())
				.build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		// record score once every 100 iterations
		model.setListeners(new ScoreIterationListener(100));

		return model;
	}


	static public MultiLayerNetwork buildModel000() throws IOException {
//		float[][] adjacent = adjacent();
		int seed = 6; // number used to initialize a pseudorandom number generator.
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1)).seed(seed)
//				.updater(new Adam(0.01)).seed(seed)// .dropOut(0.5)
				.l2(1e-4).weightInit(WeightInit.XAVIER).list()
				.layer(new DynamicGCNLayerConf.Builder().nIn(3).nOut(3).activation(Activation.RELU).build())
				.layer(new DynamicGCNLayerConf.Builder().nIn(3).nOut(3).activation(Activation.SOFTMAX).build())
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

}
