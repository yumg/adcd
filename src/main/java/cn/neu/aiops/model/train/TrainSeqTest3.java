package cn.neu.aiops.model.train;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TrainSeqTest3 {

	private static final Logger log = LoggerFactory.getLogger(TrainSeqTest3.class);

	public static void main(String[] args) throws IOException, InterruptedException {
		SequenceRecordReader trainDataReader = new CSVSequenceRecordReader();
		trainDataReader
//				.initialize(new NumberedFileInputSplit(Util.WORK_DIR + "seq-embedding2\\" + "highload-nodes-embedding - 副本 (%d).csv", 0, 19));
				.initialize(new NumberedFileInputSplit("C:\\Usr\\dev\\idea-projs\\Graph\\data\\20-18-46@2022-02-20#StochasticExceptionGraphFactory\\seq-embedding\\NodesEmbedding - 副本 (%d).csv", 0, 19));

		int miniBatchSize = 5;
		int numLabelClasses = 2;
		DataSetIterator trainDataItr = new SequenceRecordReaderDataSetIterator(trainDataReader, miniBatchSize,
				numLabelClasses, 42);

		// Normalize the training data
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainDataItr); // Collect training data statistics
		trainDataItr.reset();

		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized
		trainDataItr.setPreProcessor(normalizer);

		/// TEST DATA
		SequenceRecordReader testDataReader = new CSVSequenceRecordReader();
		testDataReader
//				.initialize(new NumberedFileInputSplit(Util.WORK_DIR + "seq-embedding2\\" + "highload-nodes-embedding - 副本 (%d).csv", 0, 19));
				.initialize(new NumberedFileInputSplit("C:\\Usr\\dev\\idea-projs\\Graph\\data\\20-18-46@2022-02-20#StochasticExceptionGraphFactory\\seq-embedding\\NodesEmbedding - 副本 (%d).csv", 0, 19));

		DataSetIterator testDataItr = new SequenceRecordReaderDataSetIterator(testDataReader, miniBatchSize,
				numLabelClasses, 42);

		testDataItr.setPreProcessor(normalizer);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123) // Random number generator seed
																						// for improved repeatability.
																						// Optional.
				.weightInit(WeightInit.XAVIER).updater(new Nadam())
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required, but
																							// helps with this data set
				.gradientNormalizationThreshold(0.5).list()
				.layer(new LSTM.Builder().activation(Activation.TANH).nIn(42).nOut(80).build())
				.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(80).nOut(numLabelClasses).build())
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		log.info("Starting training...");
		net.setListeners(new ScoreIterationListener(20),
				new EvaluativeListener(testDataItr, 1, InvocationType.EPOCH_END)); // Print
																					// the
																					// score
  																					// function
																					// value)
																					// every
																					// 20
																					// iterations
		
//		net.setListeners(new ScoreIterationListener(20));

		int nEpochs = 40;
		net.fit(trainDataItr, nEpochs);

	}
}
