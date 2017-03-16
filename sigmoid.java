package FFNN;

public class sigmoid implements learning_function {
	/**
	 * {@inheritDoc}
	 */
	@Override
	public double fn(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double fnPrine(double x) {
		double temp = fn(x);
		return temp * (1 - temp);
	}

}
