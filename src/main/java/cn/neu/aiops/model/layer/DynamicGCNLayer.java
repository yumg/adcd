package cn.neu.aiops.model.layer;

import java.util.Arrays;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.factory.Nd4j;

import cn.neu.aiops.model.conf.DynamicGCNAdjacentProvider;
import cn.neu.aiops.model.conf.DynamicGCNLayerConf;

public class DynamicGCNLayer extends BaseLayer<DynamicGCNLayerConf> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2546783984286238020L;

	private DynamicGCNAdjacentProvider adjacentProvider;

	public DynamicGCNLayer(NeuralNetConfiguration conf, DataType dataType) {
		super(conf, dataType);
		adjacentProvider = ((DynamicGCNLayerConf) conf.getLayer()).getAdjacentProvider();
	}

	@Override
	public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public boolean isPretrainLayer() {
		return false;
	}

	@Override
	public boolean hasBias() {
		return layerConf().hasBias();
	}

	@Override
	public boolean hasLayerNorm() {
		return layerConf().hasLayerNorm();
	}

	@Override
	protected Pair<INDArray, INDArray> preOutputWithPreNorm(boolean training, boolean forBackprop,
			LayerWorkspaceMgr workspaceMgr) {

		assertInputSet(forBackprop);
		applyDropOutIfNecessary(training, workspaceMgr);
		INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);
		INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training, workspaceMgr);
		INDArray g = (hasLayerNorm() ? getParam(DefaultParamInitializer.GAIN_KEY) : null);

		INDArray input = this.input.castTo(dataType);

		// Input validation:
		if (input.rank() != 2 || input.columns() != W.rows()) {
			if (input.rank() != 2) {
				throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank "
						+ input.rank() + " array with shape " + Arrays.toString(input.shape())
						+ ". Missing preprocessor or wrong input type? " + layerId());
			}
			throw new DL4JInvalidInputException(
					"Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape())
							+ ") is invalid: does not match layer input size (layer # inputs = " + W.size(0) + ") "
							+ layerId());
		}

		INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, W.dataType(), input.size(0), W.size(1));
		input.castTo(ret.dataType()).mmuli(W, ret); // TODO Can we avoid this cast? (It sohuld be a no op if not
													// required, however)

		adjacentProvider.getAdjacent().mmuli(ret, ret);

		INDArray preNorm = ret;
		if (hasLayerNorm()) {
			preNorm = (forBackprop ? ret.dup(ret.ordering()) : ret);
			Nd4j.getExecutioner().exec(new LayerNorm(preNorm, g, ret, true, 1));
		}

		if (hasBias()) {
			ret.addiRowVector(b);
		}

		if (maskArray != null) {
			applyMask(ret);
		}

		return new Pair<>(ret, preNorm);

	}

}
