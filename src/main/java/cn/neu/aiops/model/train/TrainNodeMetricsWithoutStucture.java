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
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 正常的训练没有问题
 * 
 * 
 * @author yumin
 *
 */
public class TrainNodeMetricsWithoutStucture {

	private static Logger log = LoggerFactory.getLogger(TrainNodeMetricsWithoutStucture.class);

	public static DataSetIterator allData(int batchSize) throws IOException, InterruptedException {
		CSVRecordReader csvRecordReader = new CSVRecordReader(',');
		csvRecordReader.initialize(new FileSplit(new File(Util.WORK_DIR + "NodesMetrics2.csv")));
		Schema inputDataSchema = new Schema.Builder().addColumnLong("DateTime")
				.addColumnsFloat("Metric1", "Metric2", "Metric3").addColumnInteger("Label").build();
		TransformProcess tp = new TransformProcess.Builder(inputDataSchema).removeColumns("DateTime").build();
		return new RecordReaderDataSetIterator(new TransformProcessRecordReader(csvRecordReader, tp), batchSize, 3, 2);
	}

	public static void main(String[] args) throws Exception {
		main0();
	}
	
	public static void main1() throws IOException, InterruptedException {
		DataSetIterator iterator = allData(100);
		DataSet allData = iterator.next();
		DataSet trainingData = allData;
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainingData); // Collect the statistics (mean/stdev) from the training data. This does not
		normalizer.transform(trainingData); // Apply normalization to the training data

		MultiLayerNetwork model = buildModel();

		for (int i = 0; i < 1000; i++) {
			model.fit(trainingData);
		}

		// evaluate the model on the test set
//		Evaluation eval = new Evaluation();
//		DataSet sample = allData.sample(5);
//		INDArray output = model.output(sample.getFeatures());
//		eval.eval(sample.getLabels(), output);
//		log.info(eval.stats());

		
		DataSetIterator iterator2 = allData(100);
		DataSet allData2 = iterator2.next();
		Evaluation eval = new Evaluation();
		DataSet sample = allData2.sample(5);
		INDArray output = model.output(sample.getFeatures());
		eval.eval(sample.getLabels(), output);
		log.info(eval.stats());
	}

	public static void main0() throws IOException, InterruptedException {
		DataSetIterator iterator = allData(5);
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(iterator);

		MultiLayerNetwork model = buildModel();

		for (int i = 0; i < 100; i++) {
			while (iterator.hasNext()) {
				DataSet trainingData = iterator.next();
				normalizer.transform(trainingData);
				model.fit(trainingData);
			}
			iterator.reset();
		}

		iterator.reset();
		DataSet ds = iterator.next();
		
		normalizer.fit(ds);
		normalizer.transform(ds);

		// evaluate the model on the test set
		Evaluation eval = new Evaluation();
		INDArray output = model.output(ds.getFeatures());
		eval.eval(ds.getLabels(), output);
		log.info(eval.stats());
	}

	public static MultiLayerNetwork buildModel() {
		final int numInputs = 3;
		int outputNum = 2;
		long seed = 6;

		log.info("Build model....");
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

}
