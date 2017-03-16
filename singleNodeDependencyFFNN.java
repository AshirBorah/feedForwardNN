package FFNN;

import java.util.Random;

/**
 * This class implements a neural network which has an input and output layer.
 * It does not have any hidden layers
 * 
 * @author Ashir Borah
 * @version December 16, 2016
 *
 */

public class singleNodeDependencyFFNN extends ffNeuralNetwork {
	/*
	 * Please note: even though this class has a hidden layer, this later is the
	 * predicted output layer because it is the only single layer for prediction
	 * used in the neural net.
	 */

	private static double[][] input;
	private static double[] output;

	public static void main(String args[]) throws Exception {
		Random rnd = new Random();

		initialize();

		// function selection
		learning_function fn = new reLU();

		// weight initialization
		double[] weights = new double[input[0].length];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextGaussian();
		}

		// input layer clone
		double[][] inputL = input.clone();

		for (int i = 0; i < 100000; i++) {

			// construct hidden layer
			double[] hLayer = multiply(inputL, weights);

			// use the function to flatten the values
			fnMatrix(hLayer, fn);

			// calculate the error matrix
			double[] error = matrixDiff(output, hLayer);

			// the del matrix
			double[] del = corrTerms(hLayer, error, fn);

			// correct the network
			correctNetwork(inputL, weights, del, 0.01);

			// print the progress to see if the network is learning
			if (i % 10000 == 0) {
				if (i == 0) {
					System.out.println("Error at iteration:");
				}
				System.out.println(i + " : " + errorAvg(error));
			}
		}
		// recalculating the values for the final answer

		// the predicted answers
		double[] hLayer = multiply(input, weights);

		// applying the threshold function to the predicted values
		double[] arrayTh = arrayThreshold(hLayer, threshold(hLayer), 1, 0);

		// printing the output
		printOutput(inputL, hLayer, output, arrayTh);

	}

	/**
	 * A function to initialize the input and the output. The required input and
	 * output should be uncommented. New input/output pairs should be added in
	 * the same format as the ones that are supplied.
	 */
	private static void initialize() {

		// Input A,B
		// Output ~B
		input = new double[][] { { 1, 0 }, { 1, 0 }, { 1, 1 }, { 1, 1 } };
		output = new double[] { 1, 1, 0, 0 };

		// // Input A,B
		// // Output A
		// input = new double[][] { { 1, 0 }, { 0, 0 }, { 0, 1 }, {
		// 0, 1 } };
		// output = new double[] { 1, 0, 0, 0 };
	}
}