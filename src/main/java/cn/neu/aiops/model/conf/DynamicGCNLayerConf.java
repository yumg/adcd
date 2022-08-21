package cn.neu.aiops.model.conf;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import cn.neu.aiops.model.layer.DynamicGCNLayer;

/**
 * Dense layer: a standard fully connected feed forward layer
 */
public class DynamicGCNLayerConf extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6745077055450506584L;
	private boolean hasLayerNorm = false;
	private boolean hasBias = true;
	private long numParams;

	private DynamicGCNAdjacentProvider adjacentProvider;
	
	public DynamicGCNLayerConf() {
		
	}

	private DynamicGCNLayerConf(Builder builder) {
		super(builder);
		this.hasBias = builder.hasBias;
		this.hasLayerNorm = builder.hasLayerNorm;
		this.adjacentProvider = builder.getAdjacentProvider();

		initializeConstraints(builder);
	}

	public DynamicGCNAdjacentProvider getAdjacentProvider() {
		return adjacentProvider;
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
		LayerValidation.assertNInNOutSet("DenseLayer", getLayerName(), layerIndex, getNIn(), getNOut());

		Layer ret = new DynamicGCNLayer(conf, networkDataType);
		ret.setListeners(trainingListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return DefaultParamInitializer.getInstance();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		InputType outputType = getOutputType(-1, inputType);

		numParams = initializer().numParams(this);
		long updaterStateSize = (int) getIUpdater().stateSize(numParams);

		int trainSizeFixed = 0;
		int trainSizeVariable = 0;
		if (getIDropout() != null) {
//			if (false) {
//				// TODO drop connect
//				// Dup the weights... note that this does NOT depend on the minibatch size...
//				trainSizeVariable += 0; // TODO
//			} else {
				// Assume we dup the input
				trainSizeVariable += inputType.arrayElementsPerExample();
//			}
		}

		// Also, during backprop: we do a preOut call -> gives us activations size equal
		// to the output size
		// which is modified in-place by activation function backprop
		// then we have 'epsilonNext' which is equivalent to input size
		trainSizeVariable += outputType.arrayElementsPerExample();

		return new LayerMemoryReport.Builder(layerName, DynamicGCNLayerConf.class, inputType, outputType)
				.standardMemory(numParams, updaterStateSize).workingMemory(0, 0, trainSizeFixed, trainSizeVariable) // No
																													// additional
																													// memory
																													// (beyond
																													// activations)
																													// for
																													// inference
				.cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) // No caching in
																									// DenseLayer
				.build();
	}

	public boolean hasBias() {
		return hasBias;
	}

	public boolean hasLayerNorm() {
		return hasLayerNorm;
	}

	public static class Builder extends FeedForwardLayer.Builder<Builder> {

		/**
		 * If true (default): include bias parameters in the model. False: no bias.
		 *
		 */
		private boolean hasBias = true;

		private DynamicGCNAdjacentProvider adjacentProvider;

		/**
		 * If true (default): include bias parameters in the model. False: no bias.
		 *
		 * @param hasBias If true: include bias parameters in this model
		 */
		public Builder hasBias(boolean hasBias) {
			this.setHasBias(hasBias);
			return this;
		}

		/**
		 * If true (default = false): enable layer normalization on this layer
		 *
		 */
		private boolean hasLayerNorm = false;

		public Builder hasLayerNorm(boolean hasLayerNorm) {
			this.hasLayerNorm = hasLayerNorm;
			return this;
		}

		@Override
		@SuppressWarnings("unchecked")
		public DynamicGCNLayerConf build() {
			return new DynamicGCNLayerConf(this);
		}

		public boolean isHasBias() {
			return hasBias;
		}

		public void setHasBias(boolean hasBias) {
			this.hasBias = hasBias;
		}

		public boolean isHasLayerNorm() {
			return hasLayerNorm;
		}

		public void setHasLayerNorm(boolean hasLayerNorm) {
			this.hasLayerNorm = hasLayerNorm;
		}

		public DynamicGCNAdjacentProvider getAdjacentProvider() {
			return adjacentProvider;
		}

		public Builder setAdjacentProvider(DynamicGCNAdjacentProvider adjacentProvider) {
			this.adjacentProvider = adjacentProvider;
			return this;
		}

	}

}
