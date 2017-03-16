package FFNN;

import java.util.Arrays;
import java.util.Random;

/**
 * This class implements a neural network with back propagation having two
 * layers
 * 
 * @author Ashir Borah
 * @version December 16, 2016
 */
public class TwoNodeDependencyFFNN extends ffNeuralNetwork {

	private static double[][] input;
	private static double[] output;

	public static void main(String args[]) throws Exception {
		Random rnd = new Random();
		initialize();
		// Declaring the weight layers
		double[][] weights1 = new double[input[0].length][input.length];
		double[] weights2 = new double[input.length];

		// selecting the learning function. Any function implementing the
		// learning_fucntion interface can be used
		learning_function fn = new reLU();

		// initializing the weight layers to random values which are normally
		// distributed
		for (int i = 0; i < weights1.length; i++) {
			for (int j = 0; j < weights1[0].length; j++) {
				weights1[i][j] = rnd.nextGaussian();
			}
		}
		for (int i = 0; i < weights2.length; i++) {
			weights2[i] = rnd.nextGaussian();
		}

		// cloning the input values to prevent any changes to them
		double[][] inputL = input.clone();

		// starting training
		for (int i = 0; i < 100000; i++) {

			/*
			 * forward propagation calculate the predicted values apply the
			 * learning function to the predicted values. This is done for each
			 * layer
			 */
			double[][] hLayer1 = multiply(inputL, weights1);

			fnMatrix(hLayer1, fn);

			double[] hLayer2 = multiply(hLayer1, weights2);

			fnMatrix(hLayer2, fn);

			// back propagation
			// calculate the error and correction term for each layer
			double[] error2 = matrixDiff(output, hLayer2);

			double[] delL2 = corrTerms(hLayer2, error2, fn);

			double[][] error1 = multiply(vectorToMatrix(delL2), transpose(weights2));

			double[][] delL1 = corrTerms(hLayer1, error1, fn);
			correctNetwork(hLayer1, weights2, delL2, 0.01);
			correctNetwork(inputL, weights1, delL1, 0.01);

			// print error to check progress
			if (i % 10000 == 0) {
				if (i == 0) {
					System.out.println("Error at iteration:\n");
				}
				System.out.println(errorAvg(error2));
			}
		}

		// calculate the values for the final answer
		double[][] hLayer1 = multiply(inputL, weights1);
		fnMatrix(hLayer1, fn);

		double[] hLayer2 = multiply(hLayer1, weights2);
		fnMatrix(hLayer2, fn);

		System.out.println("\nFinal output:");
		System.out.println(Arrays.toString(hLayer2) + "\n");
		double[] arrayTh = arrayThreshold(hLayer2, threshold(hLayer2), 1, 0);
		printOutput(inputL, hLayer2, output, arrayTh);
	}

	/**
	 * A function to initialize the input and the output. The required input and
	 * output should be uncommented. New input/output pairs should be added in
	 * the same format as the ones that are supplied.
	 */
	private static void initialize() {
		// data set 1
		// XOR Gate
		input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		output = new double[] { 1, 0, 0, 1 };

		// // data set 2
		// // AND Gate
		// input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		// output = new double[] { 0, 0, 0, 1 };
		//
		// // data set 3
		// // OR Gate
		// input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		// output = new double[] { 0, 1, 1, 1 };
		//
		// // data set 4
		// // NOR Gate
		// input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		// output = new double[] { 1, 0, 0, 0 };
		//
		// // data set 5
		// // NAND Gate
		// input = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		// output = new double[] { 1, 1, 1, 0 };
		//
		// // data set 6
		// Distribution of points which are separable by x=y
		// double[][] input = new double[][] { { 3, 2 }, { 6, 5 }, { 10, 2 }, {
		// 2, 1 }, { 20, 1 }, { 0.2, 1 }, { 1, 2 },
		// { 2, 3 }, { 1, 10.5 }, { 4, 5 } };
		// double[] output = new double[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
	}
}