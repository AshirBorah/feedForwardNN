package FFNN;

import java.util.Arrays;

/**
 * A class which implements the general functionality needed for implementing
 * the neural nets that the project aims to target
 * 
 * @author Ashir Borah
 * @version December 16, 2016
 *
 */

public class ffNeuralNetwork {
	/*
	 * Vectors in this project refer to 1 dimensional array. It is used
	 * synonymously for row vector and column vector.
	 * 
	 * Elementary knowledge of linear algebra is assumed.
	 */

	/**
	 * Returns an array that has been adjusted to the threshold value. The
	 * threshold function used here is the modified sign function
	 * 
	 * f(x)=type1 if x{@code>}threshold, f(x){@code=}type2 otherwise
	 * 
	 * @param arr
	 *            the input array
	 * @param threshold
	 *            the threshold value
	 * @param type1
	 *            value of x when x{@code>}threshold
	 * @param type2
	 *            value of x when x{@code<=}threshold
	 * @return modified array
	 */
	public static double[] arrayThreshold(double[] arr, double threshold, double type1, double type2) {
		double[] output = new double[arr.length];
		for (int i = 0; i < arr.length; i++) {
			if ((arr[i]) > threshold) {
				output[i] = type1;
			} else {
				output[i] = type2;
			}
		}
		return output;
	}

	/**
	 * Returns the element wise difference of two matrix C=array1-array2
	 * 
	 * @param array1
	 *            input array array1
	 * @param array2
	 *            input array array1
	 * @throws matrixException
	 *             when there is size mismatch
	 * @return the difference of array1-array2
	 */

	public static double[] matrixDiff(double[] array1, double[] array2) {
		if (array1.length != array2.length || array1.length == 0) {
			throw new matrixException("Bad input matrix for matrix difference");
		} else {
			double[] output = new double[array1.length];
			for (int i = 0; i < array1.length; i++) {
				output[i] = array1[i] - array2[i];
			}
			return output;
		}
	}

	/**
	 * Method to update the weights of a layer with the given update values.
	 * This is the version for a vector weights input
	 * 
	 * @param input
	 *            the inputs for the layer
	 * @param weights
	 *            the weights of the layer
	 * @param updates
	 *            the updates for the layer
	 * @param learningRate
	 *            the rate for learning. Lesser the number, more accurate but
	 *            slower the learning is. Usual amount is around 0.01
	 */
	public static void correctNetwork(double[][] input, double[] weights, double[] updates, double learningRate) {
		double[][] temp = transpose(input);
		double[] adds = multiply(temp, updates);
		for (int i = 0; i < adds.length; i++) {
			weights[i] += adds[i] * learningRate;
		}
	}

	/**
	 * Works just like
	 * {@link ffNeuralNetwork#correctNetwork(double[][], double[][], double[][], double)}
	 * but has the input and weight parameters in vector form
	 * 
	 */
	public static void correctNetwork(double[][] input, double[][] weights, double[][] updates, double learningRate) {
		double[][] temp = transpose(input);
		double[][] adds = multiply(temp, updates);
		for (int i = 0; i < adds.length; i++) {
			for (int j = 0; j < adds[0].length; j++) {
				weights[i][j] += adds[i][j] * learningRate;
			}
		}
	}

	/**
	 * Calculates the amount by which the weights need to be adjusted for a
	 * given layer
	 * 
	 * @param input
	 *            the input layer
	 * @param error
	 *            the error for the layer
	 * @param function
	 *            the learning function used
	 * @return the amount for each layer by which they need to be adjusted by
	 */

	public static double[] corrTerms(double[] input, double[] error, learning_function function) {
		double[] terms = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			terms[i] = error[i] * function.fnPrine(input[i]);
		}
		return terms;
	}

	/**
	 * 
	 * Works just like
	 * {@link ffNeuralNetwork#corrTerms(double[], double[], learning_function)}
	 * but has the input and weight parameters in vector form
	 */

	public static double[][] corrTerms(double[][] calc, double[][] error, learning_function function) {
		double[][] terms = new double[calc.length][calc[0].length];
		for (int i = 0; i < calc.length; i++) {
			for (int j = 0; j < calc[i].length; j++) {
				terms[i][j] = error[i][j] * function.fnPrine(calc[i][j]);
			}
		}
		return terms;
	}

	/**
	 * Calculated the absolute average error from the error vector
	 * 
	 * @param arr
	 *            the input vector
	 * @return the average error
	 */
	public static double errorAvg(double[] arr) {
		double sum = 0;
		for (int i = 0; i < arr.length; i++) {
			sum += Math.abs(arr[i]);
		}
		return sum / arr.length;
	}

	/**
	 * Prints out the final answer in a formatted way for easier interpretation
	 * 
	 * @param input
	 *            the input to the neural network
	 * @param hLayer
	 *            the final output layer for the neural net
	 * @param output
	 *            the actual output for the neural net
	 * @param arrayTh
	 *            output array after threshold function was applied
	 */
	public static void printOutput(double[][] input, double[] hLayer, double[] output, double[] arrayTh) {
		System.out.println("\nFinal Output:\nInput\t\t\tPredicted\tThreshold Output\tActual Output\t\tCorrect");
		for (int i = 0; i < input.length; i++) {
			String s = String.format("%.10f", hLayer[i]);// formats the
															// prediction to 10
															// decimal places
			s += "\t\t" + String.format("%.1f", arrayTh[i]);// formats the
			// prediction to 10
			// decimal places
			System.out.print(Arrays.toString(input[i]) + "\t\t" + s + "\t\t" + output[i] + "\t\t\t");
			if (output[i] == arrayTh[i]) {
				System.out.println("[Y]");
			} else {
				System.out.println("[x]");
			}
		}
	}

	/**
	 * A function which applies the given function to the vector.
	 * 
	 * @param arr
	 *            the input vector
	 * @param function
	 *            the function to be applied to the vector. The function needs
	 *            to implement the FFNN.learning_function interface
	 */
	public static void fnMatrix(double[] arr, learning_function function) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = function.fn(arr[i]);
		}
	}

	/**
	 * Works exactly like
	 * {@link FFNN.ffNeuralNetwork#fnMatrix(double[], learning_function)} but
	 * takes in a matrix as the input argument
	 * 
	 */
	public static void fnMatrix(double[][] arr, learning_function function) {
		for (int i = 0; i < arr.length; i++) {
			fnMatrix(arr[i], function);
		}
	}

	/**
	 * A function for dot multiplication of two vectors which results in a
	 * double. It follows the general rules of linear algebra.
	 * 
	 * @throws matrixException
	 *             when the arrays are not of equal size or either one is size 0
	 * @param array1
	 *            vector 1
	 * @param array2
	 *            vector 2
	 * @return the resultant of vector 1 (dot) vector 2
	 */
	public static double multiply(double[] array1, double[] array2) {
		if (array1.length != array2.length || array1.length == 0) {
			throw new matrixException("Bad input matrix for dot Product " + array1.length + "  " + array2.length);
		} else {
			double c = 0;
			for (int i = 0; i < array1.length; i++) {
				c += array1[i] * array2[i];
			}
			return c;
		}
	}

	/**
	 * Works like {@link FFNN.ffNeuralNetwork#multiply(double[], double[])} but
	 * one a matrix and a vector which results in a vector. This function
	 * calculates the dot product on each row.
	 * 
	 * @throws matrixException
	 *             when the matrix do not have the matching sizes for the
	 *             required operation
	 * 
	 * @param a
	 *            input matrix
	 * @param b
	 *            input vector
	 * @return vector a(dot)b
	 */
	public static double[] multiply(double[][] a, double[] b) {
		double[] c = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			c[i] = multiply(a[i], b);
		}
		return c;
	}

	/**
	 * The function carries out vector multiplication of given matrices AxB
	 * 
	 * @param a
	 *            input matrix A
	 * @param b
	 *            input matrix B
	 * @return resultant matrix AxB
	 */
	public static double[][] multiply(double[][] a, double[][] b) {
		double[][] c = new double[a.length][b[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < b[0].length; j++) {
				for (int k = 0; k < a[0].length; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return c;
	}

	/**
	 * The function converts a vector to a matrix for element wise
	 * multiplication
	 * 
	 * @param arr
	 *            input vector
	 * @return output matrix
	 */
	public static double[][] vectorToMatrix(double[] arr) {
		double[][] out = new double[arr.length][1];
		for (int i = 0; i < arr.length; i++) {
			out[i][0] = arr[i];
		}
		return out;
	}

	/**
	 * Prints a matrix in grid form
	 * 
	 * @param a
	 *            matrix to print
	 */
	public static void print2DArray(double[][] a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				System.out.print(a[i][j] + "  ");
			}
			System.out.println();
		}
		System.out.println();
	}

	/**
	 * Calculates the threshold value as the average of the min and the max
	 * elements in an array
	 * 
	 * @param a
	 *            the input vector to calculate the average of
	 * @return the threshold value of the vectors
	 */
	public static double threshold(double[] a) {
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
			}
			if (a[i] < min) {
				min = a[i];
			}
		}
		return (max + min) / 2;
	}

	/**
	 * Returns the transpose of the given vector
	 * 
	 * @param a
	 *            input vector
	 * @return the transpose of the input vector
	 */
	public static double[][] transpose(double[] a) {
		double[][] b = new double[1][a.length];
		for (int i = 0; i < a.length; i++) {
			b[0][i] = a[i];
		}
		return b;
	}

	/**
	 * Returns the transpose of the given
	 * 
	 * @param a
	 *            input matrix
	 * @return transpose of the input matrix
	 */
	public static double[][] transpose(double[][] a) {
		double[][] b = new double[a[0].length][a.length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}
}
