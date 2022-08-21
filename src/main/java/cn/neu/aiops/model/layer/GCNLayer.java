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

import cn.neu.aiops.model.conf.GCNLayerConf;
import cn.neu.aiops.model.train.Util;

/**
 */
public class GCNLayer extends BaseLayer<GCNLayerConf> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -773036181968978807L;

	private INDArray adjMatrix_ndarray;

	public GCNLayer(NeuralNetConfiguration conf, DataType dataType) {
		super(conf, dataType);
		float[][] adjMatrix_raw = ((GCNLayerConf) conf.getLayer()).getAdjMatrix();
		adjMatrix_ndarray = Util.normalize(Nd4j.create(adjMatrix_raw));
	}

	@Override
	public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
		/*
		 * The activate method is used for doing forward pass. Note that it relies on
		 * the pre-output method; essentially we are just applying the activation
		 * function (or, functions in this example). In this particular (contrived)
		 * example, we have TWO activation functions - one for the first half of the
		 * outputs and another for the second half.
		 */
		INDArray z = preOutput(training, workspaceMgr);
		INDArray ret = layerConf().getActivationFn().getActivation(z, training);

		if (maskArray != null) {
			applyMask(ret);
		}

		return ret;
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
		input.castTo(ret.dataType()).mmuli(W, ret); // TODO Can we avoid this cast? (It should be a no op if not
													// required, however)
		adjMatrix_ndarray.mmuli(ret, ret);

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

	@Override
	public boolean isPretrainLayer() {
		return false;
	}

}
