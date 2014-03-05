package com.gavinmhackeling;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream.GetField;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

/**
 * @author gavin
 * 
 * d1 = "i ate pizza"
 * d2 = "i ate cake"
 * d3 = "cats are fluffy"
 * d4 = "cats are fuzzy"
 * 
 * d5 = "cats are cats"
 * d6 = "i ate pizza cake"
 * 
 * 1 i
 * 2 ate
 * 3 pizza
 * 4 cake
 * 5 cats
 * 6 are
 * 7 fluffy
 * 8 fuzzy
 *
 */
public class LiblinearLogisticRegression {

	static int NUM_INSTANCES = 4;

	// 1 = square, -1 = non-square
	static double[] Y_TRAIN = {1, 1, -1, -1};

	/*
	 * FeatureNodes can be sparse 
	 */
	static FeatureNode[] instance1 = {new FeatureNode(1, 1), new FeatureNode(2, 1), new FeatureNode(3, 1)};
	static FeatureNode[] instance2 = {new FeatureNode(1, 1), new FeatureNode(2, 1), new FeatureNode(4, 1)};
	static FeatureNode[] instance3 = {new FeatureNode(5, 1), new FeatureNode(6, 1), new FeatureNode(7, 1)};
	static FeatureNode[] instance4 = {new FeatureNode(5, 1), new FeatureNode(6, 1), new FeatureNode(8, 1)};

	static FeatureNode[][] X_TRAIN = {instance1, instance2, instance3, instance4};


	/**
	 * @param fileName
	 * @param model
	 * @throws IOException
	 */
	public static void saveModel(String fileName, Model model) throws IOException {
		File modelFile = new File(fileName);
		model.save(modelFile);
	}

	/**
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	public static Model loadModel(String fileName) throws IOException {
		File modelFile = new File(fileName);
		return Model.load(modelFile);
	}

	public static FeatureNode[] getFeatureNodes(Map<Integer, Integer> features) {
		int numFeatures = features.keySet().size();
		FeatureNode[] n = new FeatureNode[numFeatures];
		int counter = 0;
		for (Integer key: features.keySet()) {
			n[counter] = new FeatureNode(key, features.get(key));
			counter++;
		}
		return n;
	}
	
	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		long start = System.nanoTime();
		Problem problem = new Problem();
		problem.l = NUM_INSTANCES;

		String[] documents = {"i ate pizza", "i ate cake", "cats are fluffy", "cats are fuzzy"};
		
		Map<String, Integer> dictionary = new HashMap<String, Integer>(50);
		
		int index = 1;
		for (String document: documents) {
			String[] tokens = document.split(" ");
			for (String token: tokens) {
				if (!dictionary.containsKey(token)) {
					dictionary.put(token, index);
					index++;
				}
			}
		}
		
//		FeatureNode[] testInstance1 = {new FeatureNode(5, 2), new FeatureNode(6, 1)};
		List<FeatureNode[]> instances = new ArrayList<FeatureNode[]>();
		for (String document: documents) {
			String tokens[] = document.split(" ");
			Map<Integer, Integer> features = new HashMap<Integer, Integer>();
			for (String token: tokens) {
				int i = dictionary.get(token);
				features.put(i, 1); // change the value to a counter
			}
			instances.add(getFeatureNodes(features));
		}
		
		for (FeatureNode[] instance: instances) {
			System.out.println("Instance: " + instance);
		}
		
		
//		Map<String, Counter> counts = 
		// number of features
		problem.n = 8;
		problem.x = X_TRAIN;
		problem.y = Y_TRAIN;

		SolverType solver = SolverType.L2R_LR;
		double C = 1.0;
		double eps = 0.01;

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);

		long start2 = System.nanoTime();

		// target = -1
		FeatureNode[] testInstance1 = {new FeatureNode(5, 2), new FeatureNode(6, 1)};
		FeatureNode[] testInstance2 = {new FeatureNode(1, 1), new FeatureNode(2, 1), new FeatureNode(3, 1), new FeatureNode(4, 1)};

		double prediction = Linear.predict(model, testInstance1);
		double[] probabilityEstimates = {0.0, 1.0};
		double prob = Linear.predictProbability(model, testInstance1, probabilityEstimates);
		double prob2 = Linear.predictProbability(model, testInstance2, probabilityEstimates);
		System.out.println("prob1: " + prob);
		System.out.println("prob2: " + prob2);
		System.out.println("Predicted label: " + prediction);
		float predictionTime = (System.nanoTime() - start2) / 1000 / 1000;
		System.out.println("Predicted in: " + predictionTime + " milliseconds.");
		System.out.println("Completed in: " + (System.nanoTime() - start) / 1000 / 1000 + " milliseconds.");
	}

}
