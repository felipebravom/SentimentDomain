package tsa.core;

import org.apache.commons.math.DimensionMismatchException;
import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.NotPositiveDefiniteMatrixException;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.random.CorrelatedRandomVectorGenerator;
import org.apache.commons.math.random.GaussianRandomGenerator;
import org.apache.commons.math.random.JDKRandomGenerator;
import org.apache.commons.math.random.RandomGenerator;

public class RandomNumbersExample {
	
	static public void main(String args[]) throws NotPositiveDefiniteMatrixException, DimensionMismatchException{
		
		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(17399225432l);  // Fixed seed means same results every time

		// Create a GassianRandomGenerator using rg as its source of randomness
		GaussianRandomGenerator rawGenerator = new GaussianRandomGenerator(rg);

		double[] mean = {0, 0};
		double[][] cov = {{5, 0}, {0, 5}};
		RealMatrix covariance = MatrixUtils.createRealMatrix(cov); 
		
		
		// Create a CorrelatedRandomVectorGenerator using rawGenerator for the components
		CorrelatedRandomVectorGenerator generator =    new CorrelatedRandomVectorGenerator(mean, covariance, 1.0e-12 * covariance.getNorm(), rawGenerator);

		// Use the generator to generate correlated vectors
	
		
		for(int i=0;i<100;i++){
			double[] randomVector = generator.nextVector();
			for( double d:randomVector){
				System.out.println(d);
			}
			System.out.println("--------");
			
		}

		
		
	}

}
