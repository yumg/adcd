package cn.neu.aiops.model.conf;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import cn.neu.aiops.model.layer.GCNLayer;

/**
 *
 */
public class GCNLayerConf extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -565420660076120372L;

	private float[][] adjMatrix;

	public GCNLayerConf() {
		// We need a no-arg constructor so we can deserialize the configuration from
		// JSON or YAML format
		// Without this, you will likely get an exception like the following:
		// com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor
		// found for type [simple type, class
		// org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not
		// instantiate from JSON object (missing default constructor or creator, or
		// perhaps need to add/enable type information?)
	}

	public float[][] getAdjMatrix() {
		return adjMatrix;
	}

	private GCNLayerConf(Builder builder) {
		super(builder);
		adjMatrix = builder.adj;
		initializeConstraints(builder);
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDType) {
		// The instantiate method is how we go from the configuration class (i.e., this
		// class) to the implementation class
		// (i.e., a CustomLayerImpl instance)
		// For the most part, it's the same for each type of layer

		GCNLayer gcnLayerImpl = new GCNLayer(conf, networkDType);
		gcnLayerImpl.setListeners(iterationListeners); // Set the iteration listeners, if any
		gcnLayerImpl.setIndex(layerIndex); // Integer index of the layer

		// Parameter view array: In Deeplearning4j, the network parameters for the
		// entire network (all layers) are
		// allocated in one big array. The relevant section of this parameter vector is
		// extracted out for each layer,
		// (i.e., it's a "view" array in that it's a subset of a larger array)
		// This is a row vector, with length equal to the number of parameters in the
		// layer
		gcnLayerImpl.setParamsViewArray(layerParamsView);

		// Initialize the layer parameters. For example,
		// Note that the entries in paramTable (2 entries here: a weight array of shape
		// [nIn,nOut] and biases of shape [1,nOut]
		// are in turn a view of the 'layerParamsView' array.
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		gcnLayerImpl.setParamTable(paramTable);
		gcnLayerImpl.setConf(conf);
		return gcnLayerImpl;
	}

	@Override
	public ParamInitializer initializer() {
		// This method returns the parameter initializer for this type of layer
		// In this case, we can use the DefaultParamInitializer, which is the same one
		// used for DenseLayer
		// For more complex layers, you may need to implement a custom parameter
		// initializer
		// See the various parameter initializers here:
		// https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/params

		return DefaultParamInitializer.getInstance();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		// Memory report is used to estimate how much memory is required for the layer,
		// for different configurations
		// If you don't need this functionality for your custom layer, you can return a
		// LayerMemoryReport
		// with all 0s, or

		// This implementation: based on DenseLayer implementation
		InputType outputType = getOutputType(-1, inputType);

		long numParams = initializer().numParams(this);
		int updaterStateSize = (int) getIUpdater().stateSize(numParams);

		int trainSizeFixed = 0;
		int trainSizeVariable = 0;
		if (getIDropout() != null) {
			// Assume we dup the input for dropout
			trainSizeVariable += inputType.arrayElementsPerExample();
		}

		// Also, during backprop: we do a preOut call -> gives us activations size equal
		// to the output size
		// which is modified in-place by activation function backprop
		// then we have 'epsilonNext' which is equivalent to input size
		trainSizeVariable += outputType.arrayElementsPerExample();

		return new LayerMemoryReport.Builder(layerName, GCNLayerConf.class, inputType, outputType)
				.standardMemory(numParams, updaterStateSize).workingMemory(0, 0, trainSizeFixed, trainSizeVariable)
				.cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS).build();
	}

	// Here's an implementation of a builder pattern, to allow us to easily
	// configure the layer
	// Note that we are inheriting all of the FeedForwardLayer.Builder options:
	// things like n
	public static class Builder extends FeedForwardLayer.Builder<Builder> {

		private float[][] adj;

		// This is an example of a custom property in the configuration

		/**
		 * A custom property used in this custom layer example. See the README.md for
		 * details
		 *
		 * @param secondActivationFunction Second activation function for the layer
		 */

		public Builder adjacentMatrix(float[][] adj) {
			this.adj = adj;
			return this;
		}

		@Override
		@SuppressWarnings("unchecked") // To stop warnings about unchecked cast. Not required.
		public GCNLayerConf build() {
			return new GCNLayerConf(this);
		}
	}

}
