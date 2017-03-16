package FFNN;

public class tanh implements learning_function {
	/**
	 * {@inheritDoc}
	 */
	@Override
	public double fn(double x) {
		return 1 / (1 + Math.tanh(x));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double fnPrine(double x) {
		double temp = Math.cosh(x);
		temp = temp * temp;
		return 1 / temp;
	}

}
