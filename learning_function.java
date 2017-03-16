package FFNN;

public interface learning_function {
	/*
	 * This interface is used because of lack of function pointers in Java Every
	 * class that is used as the learning function implements this interface so
	 * that we can use the sub type principle to maintain cleaner code.
	 */
	/**
	 * The learning function
	 * 
	 * @param x
	 *            the input value for the function
	 * @return the result of the function
	 */
	double fn(double x);

	/**
	 * The derivative of the function fn
	 * 
	 * @param x
	 *            the value at which the derivative is supposed to be calculated
	 * @return the value of the derivative at x
	 */
	double fnPrine(double x);
}
