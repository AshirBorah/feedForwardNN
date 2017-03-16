package FFNN;

public class reLU implements learning_function {
	/**
	 * {@inheritDoc}
	 * 
	 */
	@Override
	public double fn(double x) {
		return Math.log(1 + Math.exp(x));
	}

	/**
	 * {@inheritDoc}
	 * 
	 */
	@Override
	public double fnPrine(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}
