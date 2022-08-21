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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public class ModelXORTEST4 {

	private static final Logger log = LoggerFactory.getLogger(ModelXORTEST4.class);

	public static void main(String[] args) throws IOException, InterruptedException {
		log.info("Data preparation...");

		CSVRecordReader csvRecordReader = new CSVRecordReader();
		csvRecordReader.initialize(new FileSplit(new File("C:\\Usr\\data.csv")));

		DataSetIterator iterator = new RecordReaderDataSetIterator(csvRecordReader, 6, 6, 2);
		DataSet ds = iterator.next();

		log.info("Features:\n {}", ds.getFeatures());
		log.info("Labels:\n {}", ds.getLabels());

		log.info("Network configuration and training...");

		MultiLayerNetwork net = MultiLayerNetwork.load(new File("c:\\Usr\\model.txt"), true);
		// add an listener which outputs the error every 100 parameter updates

		// let Evaluation prints stats how often the right output had the highest value

		INDArray output = net.output(ds.getFeatures());

		System.out.println(output);

		Evaluation eval = new Evaluation();
		eval.eval(ds.getLabels(), output);
		System.out.println(eval.stats());

	}
}
