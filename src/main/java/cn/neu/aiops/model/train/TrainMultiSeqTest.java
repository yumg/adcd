package cn.neu.aiops.model.train;

import java.io.IOException;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainMultiSeqTest {

//	private static final Logger log = LoggerFactory.getLogger(TrainMultiSeqTest.class);

	public static void main(String[] args) throws IOException, InterruptedException {

		RecordReaderMultiDataSetIterator trainDataItr = getTrainDataItr();
		RecordReaderMultiDataSetIterator testDataItr = getTestDataItr();

		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.01)).graphBuilder()
				.addInputs("seq1", "seq2").setInputTypes(InputType.recurrent(1), InputType.recurrent(1))
				.addLayer("LSTM1", new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build(), "seq1")
				.addLayer("LSTM2", new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build(), "seq2")
				.addVertex("merge", new MergeVertex(), "LSTM1", "LSTM2")
				.addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.activation(Activation.SOFTMAX).nIn(20).nOut(2).build(), "merge")
				.setOutputs("out").build();

		ComputationGraph net = new ComputationGraph(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(20),
				new EvaluativeListener(testDataItr, 1, InvocationType.EPOCH_END));
		int nEpochs = 100;
		net.fit(trainDataItr, nEpochs);

	}

	public static RecordReaderMultiDataSetIterator getTrainDataItr() throws IOException, InterruptedException {
		SequenceRecordReader seqReader1 = new CSVSequenceRecordReader();
		seqReader1.initialize(new NumberedFileInputSplit(GenSeqTest.dirBase + GenSeqTest.filePrefix + "%d.csv", 0, 99));

		SequenceRecordReader seqReader2 = new CSVSequenceRecordReader();
		seqReader2.initialize(new NumberedFileInputSplit(GenSeqTest.dirBase + GenSeqTest.filePrefix + "%d.csv", 0, 99));

		RecordReaderMultiDataSetIterator itr = new RecordReaderMultiDataSetIterator.Builder(5)
				.addSequenceReader("seq1", seqReader1).addSequenceReader("seq2", seqReader2).addInput("seq1", 0, 0)
				.addInput("seq2", 0, 0).addOutputOneHot("seq1", 1, 2).build();

		return itr;
	}

	public static RecordReaderMultiDataSetIterator getTestDataItr() throws IOException, InterruptedException {
		SequenceRecordReader seqReader1 = new CSVSequenceRecordReader();
		seqReader1.initialize(new NumberedFileInputSplit(GenSeqTest.dirBase + GenSeqTest.filePrefix + "%d.csv", 0, 99));

		SequenceRecordReader seqReader2 = new CSVSequenceRecordReader();
		seqReader2.initialize(new NumberedFileInputSplit(GenSeqTest.dirBase + GenSeqTest.filePrefix + "%d.csv", 0, 99));

		RecordReaderMultiDataSetIterator itr = new RecordReaderMultiDataSetIterator.Builder(5)
				.addSequenceReader("seq1", seqReader1).addSequenceReader("seq2", seqReader2).addInput("seq1", 0, 0)
				.addInput("seq2", 0, 0).addOutputOneHot("seq1", 1, 2).build();

		return itr;
	}
}
