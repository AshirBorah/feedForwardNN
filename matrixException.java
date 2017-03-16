package FFNN;

@SuppressWarnings("serial")
public class matrixException extends RuntimeException {
	/**
	 * This class is created to throw exceptions which do not allow certain
	 * matrix approximations. For eg; for element wise substraction of two
	 * vectors, they must be of the same dimension.
	 */
	public matrixException() {
	}

	public matrixException(String msg) {
		super(msg);
	}

}
