package cn.neu.aiops.model.conf;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS)
public interface DynamicGCNAdjacentProvider extends Serializable {
	INDArray getAdjacent();
}
