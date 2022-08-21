/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package cn.neu.aiops.model.train;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cn.neu.aiops.model.conf.GCNLayerConf;

/**
 * This basic example shows how to manually create a DataSet and train it to an
 * basic Network.
 * <p>
 * The network consists in 2 input-neurons, 1 hidden-layer with 4
 * hidden-neurons, and 2 output-neurons.
 * <p>
 * I choose 2 output neurons, (the first fires for false, the second fires for
 * true) because the Evaluation class needs one neuron per classification.
 * <p>
 * +---------+---------+---------------+--------------+ | Input 1 | Input 2 |
 * Label 1(XNOR) | Label 2(XOR) |
 * +---------+---------+---------------+--------------+ | 0 | 0 | 1 | 0 |
 * +---------+---------+---------------+--------------+ | 1 | 0 | 0 | 1 |
 * +---------+---------+---------------+--------------+ | 0 | 1 | 0 | 1 |
 * +---------+---------+---------------+--------------+ | 1 | 1 | 1 | 0 |
 * +---------+---------+---------------+--------------+
 *
 * @author Peter Großmann
 * @author Dariusz Zbyrad
 */
public class ModelXORTEST3 {

	private static final Logger log = LoggerFactory.getLogger(ModelXORTEST3.class);

	public static void main(String[] args) throws IOException, InterruptedException {
		INDArray adj = Nd4j.readTxt("C:\\Usr\\adj.csv");
		log.info("Adjacent~ Matrix:\n {}", adj.toString());
		float[][] adjMatrix = adj.toFloatMatrix();
//		adj = Util.normalize(adj);
//		log.info("Adjacent Matrix:\n {}", adj.toString());

		int seed = 1234; // number used to initialize a pseudorandom number generator.
		int nEpochs = 5000; // number of training epochs

		log.info("Data preparation...");
		CSVRecordReader csvRecordReader = new CSVRecordReader();
		csvRecordReader.initialize(new FileSplit(new File("C:\\Usr\\data.csv")));

		DataSetIterator iterator = new RecordReaderDataSetIterator(csvRecordReader, 6, 6, 2);
		DataSet ds = iterator.next();
		
		log.info("Features:\n {}", ds.getFeatures());
		log.info("Labels:\n {}", ds.getLabels());


		log.info("Network configuration and training...");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .updater(new Sgd(0.1))
				.updater(new Adam(0.01)).seed(seed)
//            .l2(0.0005)
				.biasInit(0) // init the bias with 0 - empirical value, too
				// The networks can process the input more quickly and more accurately by
				// ingesting
				// minibatches 5-10 elements at a time in parallel.
				// This example runs better without, because the dataset is smaller than the
				// mini batch size
				.miniBatch(false).list()
				.layer(new GCNLayerConf.Builder().nIn(6).nOut(16).adjacentMatrix(adjMatrix).activation(Activation.RELU)
						// random initialize weights with values between 0 and 1
						.weightInit(new UniformDistribution(0, 1)).build())

				.layer(new GCNLayerConf.Builder().nIn(16).nOut(2).adjacentMatrix(adjMatrix).activation(Activation.SOFTMAX)
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

//		DataSet train = ds.get(new int[] {0,3});
		// here the actual learning takes place
//		ds.splitTestAndTrain(0.1);
		ds.splitTestAndTrain(1);
		for (int i = 0; i < nEpochs; i++) {
			net.fit(ds);
		}

		// create output for every training sample
		INDArray output = net.output(ds.getFeatures());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
		Evaluation eval = new Evaluation();
		eval.eval(ds.getLabels(), output);
		System.out.println(eval.stats());
		
		net.save(new File("C:\\Usr\\model.txt"));

	}
}
