package tsa.core;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class TwitterNLPwordToVector extends SimpleBatchFilter {

	
	
	
	@Override
	public String globalInfo() {
	     return   "A simple batch filter that adds an additional attribute 'bla' at the end "
	             + "containing the index of the processed instance.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat){
		Instances result = new Instances(inputFormat, 0);
	    result.insertAttributeAt(new Attribute("bla"), result.numAttributes());
	    return result;	
	}

	
	
	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(determineOutputFormat(instances), 0);
	     for (int i = 0; i < instances.numInstances(); i++) {
	       double[] values = new double[result.numAttributes()];
	       for (int n = 0; n < instances.numAttributes(); n++)
	         values[n] = instances.instance(i).value(n);
	       values[values.length - 1] = i;
	       result.add(new SparseInstance(1, values));
	     }
	     return result;
	}

}
