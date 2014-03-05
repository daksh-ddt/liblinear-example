package com.gavinmhackeling;

import java.io.File;
import java.io.IOException;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class LiblinearLogisticRegression {

	static int NUM_INSTANCES = 4;

    // 1 = square, -1 = non-square
    static double[] Y_TRAIN = {1, 1, -1, -1};

    // squares
    static FeatureNode[] instance1 = {new FeatureNode(1, 1), new FeatureNode(2, 0)};
    static FeatureNode[] instance2 = {new FeatureNode(1, 0.9), new FeatureNode(2, 0.1)};
    static FeatureNode[] instance3 = {new FeatureNode(1, 0.1), new FeatureNode(2, 1.0)};
    static FeatureNode[] instance4 = {new FeatureNode(1, 0.2), new FeatureNode(2, 0.9)};

    static FeatureNode[][] X_TRAIN = {instance1, instance2, instance3, instance4};
    
    public static void main(String[] args) throws IOException {
    	long start = System.nanoTime();
        Problem problem = new Problem();
        problem.l = NUM_INSTANCES;

        // number of features
        problem.n = 2;
        problem.x = X_TRAIN;
        problem.y = Y_TRAIN;

        SolverType solver = SolverType.L2R_LR;
        double C = 1.0;
        double eps = 0.01;

        Parameter parameter = new Parameter(solver, C, eps);
        Model model = Linear.train(problem, parameter);
//        File modelFile = new File("model");
//        model.save(modelFile);
        // load model or use it directly
//        model = Model.load(modelFile);

        long start2 = System.nanoTime();
        Feature[] instance = {new FeatureNode(1, 0.4), new FeatureNode(2, 0)};
        double prediction = Linear.predict(model, instance);
        double[] probabilityEstimates = {0.5, 0.5};
        double prob = Linear.predictProbability(model, instance, probabilityEstimates);
        System.out.println("prob: " + prob);
        System.out.println("Predicted label: " + prediction);
        float predictionTime = (System.nanoTime() - start2) / 1000 / 1000;
        System.out.println("Predicted in: " + predictionTime + " milliseconds.");
        System.out.println("Completed in: " + (System.nanoTime() - start) / 1000 / 1000 + " milliseconds.");
    }
	
}
