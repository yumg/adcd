package cn.neu.aiops.model.train;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;

public class Util {
    public static final String WORK_DIR = "c:\\Usr\\data\\";

    public static INDArray normalize(INDArray matrix) {
        INDArray sum = matrix.sum(1);
        INDArray r_inv = Transforms.pow(sum, -1);
        r_inv.replaceWhere(Nd4j.zeros(r_inv.length()), Conditions.isInfinite());
        INDArray r_mat_inv = Nd4j.diag(r_inv);
        INDArray mul = r_mat_inv.mmul(matrix);
        return mul;
    }

    public static INDArray degree(INDArray matrix) {
        INDArray degree = matrix.sum(1);
        return Nd4j.diag(degree);
    }

    public static INDArray gcnLayer(INDArray x, INDArray adj, INDArray w) {
        INDArray degreeMatrix = degree(adj);
        INDArray degreeMatrix_05 = Transforms.pow(degreeMatrix, -0.5);
        degreeMatrix_05.replaceWhere(Nd4j.zeros(degreeMatrix_05.length()), Conditions.isInfinite());
        adj = adj.add(Nd4j.eye(adj.size(0)));
        INDArray res1 = degreeMatrix_05.mmul(adj).mmul(degreeMatrix_05).mmul(x).mmul(w);
        INDArray rtv = Nd4j.nn().relu(res1, 0);
        return rtv;
    }


    public static INDArray gcnLayer2(INDArray x, INDArray adj, INDArray w) {
        INDArray degreeMatrix = degree(adj);
        INDArray degreeMatrix_1 = Transforms.pow(degreeMatrix, -1);
        degreeMatrix_1.replaceWhere(Nd4j.zeros(degreeMatrix_1.length()), Conditions.isInfinite());
        adj = adj.add(Nd4j.eye(adj.size(0)));
        INDArray res1 = degreeMatrix_1.mmul(adj).mmul(x).mmul(w);
        INDArray rtv = Nd4j.nn().relu(res1, 0);
        return rtv;
    }

    public static void main(String[] args) throws IOException, InterruptedException {
//        main0();
    }

    public static void main1() {
        INDArray x = null, adj = null, w1 = null, w2 = null;
        INDArray l1 = gcnLayer(x, adj, w1);
        INDArray l2 = gcnLayer(l1, adj, w2);
        System.out.println(l2);
    }

    public static void main0() {
        INDArray tt = Nd4j.ones(3, 3);
        tt.putScalar(1, 1, 2);
        INDArray diag_tt = Nd4j.diag(Nd4j.diag(tt));
        System.out.println("diag_tt====");
        System.out.println(diag_tt);

        INDArray invert = InvertMatrix.invert(diag_tt, false);
        System.out.println("diag_tt^-1====");
        System.out.println(invert);

        INDArray invert_left = InvertMatrix.pLeftInvert(diag_tt, false);
        System.out.println("diag_tt^-1-left===");
        System.out.println(invert_left);

        INDArray invert_right = InvertMatrix.pRightInvert(diag_tt, false);
        System.out.println("diag_tt^-1-right===");
        System.out.println(invert_right);

        INDArray pow1 = Transforms.pow(diag_tt, -1);
        System.out.println("power-1 diag_tt====");
        System.out.println(pow1);
        pow1.replaceWhere(Nd4j.zeros(pow1.length()), Conditions.isInfinite());
//        INDArray invert = InvertMatrix.invert(tt, false);
//        System.out.println(invert);
        System.out.println("power-1 diag_tt replace infinite====");
        System.out.println(pow1);
        System.out.println("====");

        INDArray pow05 = Transforms.pow(diag_tt, -0.5);
        System.out.println("power-05 diag_tt====");
        System.out.println(pow05);
        System.out.println("power-05 diag_tt replace infinite====");
        pow05.replaceWhere(Nd4j.zeros(pow05.length()), Conditions.isInfinite());
        System.out.println(pow05);
        System.out.println("====");

        INDArray zeros = Nd4j.zeros(10, 2);

        INDArray Adj = Nd4j.ones(3, 3);
        Adj.putScalar(0, 0, 0);
        Adj.putScalar(1, 1, 0);
        Adj.putScalar(2, 2, 0);
        Adj.putScalar(1, 2, 0);
        Adj.putScalar(2, 1, 0);
        System.out.println("Adj===\n" + Adj);

        INDArray X = Nd4j.ones(3, 2);
        X.putScalar(0, 0, 3);
        X.putScalar(0, 1, 4);
        X.putScalar(1, 0, 2);
        X.putScalar(1, 1, 3);
        X.putScalar(2, 0, 3);
        X.putScalar(2, 1, 5);
        System.out.println("X===\n" + X);

        Adj = Adj.add(Nd4j.eye(3));
        System.out.println("Adj+eye===\n" + Adj);

        INDArray D = Nd4j.diag(Nd4j.create(new float[]{3, 2, 2}));
        System.out.println("D===\n" + D);

        INDArray mmul = Adj.mmul(X);
        System.out.println("Adj*X====\n" + mmul);

        INDArray D1 = Transforms.pow(D, -1);
        D1.replaceWhere(Nd4j.zeros(D1.length()), Conditions.isInfinite());
        System.out.println("D1====\n" + D1);

        INDArray D1_Adj = D1.mmul(Adj);
        System.out.println("D1_Adj===\n" + D1_Adj);

        INDArray D05 = Transforms.pow(D, -0.5);
        D05.replaceWhere(Nd4j.zeros(D05.length()), Conditions.isInfinite());
        System.out.println("D05====\n" + D05);

        INDArray D05_Adj_D05 = D05.mmul(Adj).mmul(D05);
        System.out.println("D05_Adj_D05====\n" + D05_Adj_D05);

        INDArray mmul1 = D1_Adj.mmul(X);
        System.out.println("D1_Adj*X===\n" + mmul1);

        INDArray mmul2 = D05.mmul(Adj).mmul(D05).mmul(X);
        System.out.println("D05_Adj_D05*X===\n" + mmul2);

        INDArray rand = Nd4j.rand(3, 3);
        System.out.println("Random \n" + rand);


        INDArray eye = Nd4j.eye(3);
        System.out.println("Eye ========\n" + eye);

        INDArray mmul3 = D05_Adj_D05.mmul(eye);
        System.out.println("D05_Adj_D05.mmul(eye)==\n" + mmul3);


        INDArray Adj_ = Nd4j.ones(3, 3);
        Adj_.putScalar(0, 2, 0);
        Adj_.putScalar(1, 2, 0);
        Adj_.putScalar(2, 0, 0);
        Adj_.putScalar(2, 1, 0);
        System.out.println("Adj_===\n" + Adj_);

        INDArray D_ = Nd4j.diag(Nd4j.create(new float[]{2, 2, 1}));
        System.out.println("D_===\n" + D_);

        INDArray D1_ = Transforms.pow(D_, -1);
        D1_.replaceWhere(Nd4j.zeros(D1_.length()), Conditions.isInfinite());
        System.out.println("D1_====\n" + D1_);

        INDArray D1__Adj_ = D1_.mmul(Adj_);
        System.out.println("D1__Adj===\n" + D1__Adj_);

        INDArray D_05 = Transforms.pow(D_, -0.5);
        D_05.replaceWhere(Nd4j.zeros(D_05.length()), Conditions.isInfinite());
        System.out.println("D_05====\n" + D_05);

        INDArray mmul4 = D_05.mmul(Adj_).mmul(D_05).mmul(eye);
        System.out.println("D_05.mmul(Adj_).mmul(D_05).mmul(eye)===\n" + mmul4);

//        INDArray mmul3 = D05_Adj_D05.mmul(eye);
//        System.out.println("D05_Adj_D05.mmul(eye)==\n" + mmul3);

//        INDArray mmul3 = D_05.mmul(Adj_).mmul(D_05).mmul(X);
//        System.out.println("D05_Adj_D05*X===\n" + mmul3);

        INDArray relu = Nd4j.nn().relu(mmul4, 0);
        System.out.println(relu);
    }

}
