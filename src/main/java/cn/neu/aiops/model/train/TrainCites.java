package cn.neu.aiops.model.train;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import cn.neu.aiops.model.conf.GCNLayerConf;

public class TrainCites {
	public static String cites = "C:\\Users\\yumin\\Desktop\\code\\16405589-pygcn-master\\pygcn\\data\\cora\\cora.cites";
	public static String content = "C:\\Users\\yumin\\Desktop\\code\\16405589-pygcn-master\\pygcn\\data\\cora\\cora.content";

	public static Stream<String> contentLines() throws IOException {
		return Files.lines(FileSystems.getDefault().getPath(content));
	}

	public static Stream<String> citeLines() throws IOException {
		return Files.lines(FileSystems.getDefault().getPath(cites));
	}

	public static float[][] adjacent() throws IOException {
		Stream<String> contentLines = contentLines();
		List<String> nodes = contentLines.map(s -> s.substring(0, s.indexOf("\t"))).collect(Collectors.toList());

		float[][] adjacent = new float[nodes.size()][nodes.size()];

		citeLines().forEach(l -> {
			String[] split = l.split("\t");
			int i1 = nodes.indexOf(split[0]);
			int i2 = nodes.indexOf(split[1]);
			adjacent[i1][i2] = 1f;
			adjacent[i2][i1] = 1f;
		});

		for (int i = 0; i < adjacent.length; i++) {
			adjacent[i][i] = 1f;
		}

		return adjacent;
	}

	public static Set<String> labels() throws IOException {
		Set<String> labels = contentLines().map(s -> s.substring(s.lastIndexOf("\t") + 1, s.length()))
				.collect(Collectors.toSet());
		return labels;
	}

	public static DataSet allData(Set<String> labels) throws IOException, InterruptedException {
		CSVRecordReader csvRecordReader = new CSVRecordReader('\t');
		csvRecordReader.initialize(new FileSplit(new File(content)));

		Schema inputDataSchema = new Schema.Builder().addColumnString("Node").addColumnsInteger("Metric_%d", 0, 1432)
				.addColumnCategorical("Label", new ArrayList<String>(labels)).build();

		TransformProcess tp = new TransformProcess.Builder(inputDataSchema).removeColumns("Node")
				.categoricalToInteger("Label").build();

		DataSetIterator iterator = new RecordReaderDataSetIterator(
				new TransformProcessRecordReader(csvRecordReader, tp), 2708, 1433, labels.size());
		DataSet ds = iterator.next();

		return ds;
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		float[][] adjacent = adjacent();
		Set<String> labels = labels();
		DataSet ds = allData(labels);

//		DataNormalization normalizer = new NormalizerStandardize();
		DataNormalization normalizer = new NormalizerMinMaxScaler();
		normalizer.fit(ds); // Collect the statistics (mean/stdev) from the training data. This does not
							// modify the input data
		normalizer.transform(ds);

		int seed = 1234; // number used to initialize a pseudorandom number generator.
		int nEpochs = 1500; // number of training epochs

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//	            .updater(new Sgd(0.001)).seed(seed)
				.updater(new Adam(0.01)).seed(seed)// .dropOut(0.5)
//	            .l2(0.0005)
				.biasInit(0) // init the bias with 0 - empirical value, too
				// The networks can process the input more quickly and more accurately by
				// ingesting
				// minibatches 5-10 elements at a time in parallel.
				// This example runs better without, because the dataset is smaller than the
				// mini batch size
				.miniBatch(false).list()
				.layer(new GCNLayerConf.Builder().nIn(1433).nOut(16).adjacentMatrix(adjacent).activation(Activation.RELU)
						// random initialize weights with values between 0 and 1
						.weightInit(new UniformDistribution(0, 1)).build())

				.layer(new GCNLayerConf.Builder().nIn(16).nOut(7).adjacentMatrix(adjacent).activation(Activation.SOFTMAX)
						// random initialize weights with values between 0 and 1
						.weightInit(new UniformDistribution(0, 1)).build())

				.layer(new LossLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())

				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		// add an listener which outputs the error every 100 parameter updates
		net.setListeners(new ScoreIterationListener(100));

		// C&P from LSTMCharModellingExample
		// Print the number of parameters in the network (and for each layer)
		System.out.println(net.summary());

		ds.splitTestAndTrain(0.5);
		for (int i = 0; i < nEpochs; i++) {
			net.fit(ds);
		}

		INDArray output = net.output(ds.getFeatures());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
		Evaluation eval = new Evaluation();
		eval.eval(ds.getLabels(), output);
		System.out.println(eval.stats());

	}
}
