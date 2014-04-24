package tsa.core;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class SimpleBatch extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2856512126983552314L;

	@Override
	public String globalInfo() {
		return "A simple batch filter that adds an additional attribute 'bla' at the end containing the index of the processed instance.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {
		Instances result = new Instances(inputFormat, 0);
		for(int i=0;i<10;i++){
			result.insertAttributeAt(new Attribute("bla"+i), result.numAttributes());
			
		}
		
		//result.insertAttributeAt(new Attribute("bla"), result.numAttributes());
		//result.insertAttributeAt(new Attribute("bla2"), result.numAttributes());
		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(determineOutputFormat(instances), 0);
		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);
			values[values.length - 2] = i;
			values[values.length - 1] = i+1;
			result.add(new SparseInstance(1, values));
		}
		return result;
	}

}
