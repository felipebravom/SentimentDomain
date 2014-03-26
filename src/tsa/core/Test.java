package tsa.core;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Test {
	


	public static void main(String[] args) {
		
		ArrayList<Attribute> att=new ArrayList<Attribute>(); 
	
		att.add(new Attribute("att1"));
		att.add(new Attribute("att2"));
		
	
		ArrayList<String> att3Values=new ArrayList<String>();
		for(int i=0;i<5;i++)
			att3Values.add("val" + (i+1));
		att.add(new Attribute("att3",att3Values));
		
		

		att.add(new Attribute("content", (ArrayList<String>) null));
		
		
		ArrayList<String> label=new ArrayList<String>();
		label.add("positive");
		label.add("neutral");
		label.add("negative");
		
		att.add(new Attribute("class",label));	
		
		Instances dataset=new Instances("Twitter Sentiment Analysis Dataset", att,0); // The last attribute 
		
		
		
		
	     
		System.out.println(dataset.toSummaryString());
		
		
		// Add data
		double[] values = new double[dataset.numAttributes()];
		values[0] = 2.3;
		values[1] = Math.PI;
		
		values[2] = dataset.attribute("att3").indexOfValue("val3");
		
		// to retrieve the position of a certain attribute
		System.out.println("index" + dataset.attribute("att3").index());

		
		values[3] = dataset.attribute(3).addStringValue("This is a string");

		values[4] = dataset.attribute(4).indexOfValue("neutral");
		
		System.out.println(dataset.attribute("lalal")==null);
		
		// values[2] = dataset.attribute(2).indexOfValue("val3"); It is also possible to use the numeric index
	
		
		Instance inst = new DenseInstance(1, values);
		dataset.add(inst);


		
		
		double values2[] = new double[dataset.numAttributes()];
		values2[0] = 98;
		values2[1] = 122;
		
		values2[2] = dataset.attribute("att3").indexOfValue("val1");
				
		values2[3] = dataset.attribute(3).addStringValue("second string");

		values2[4] = dataset.attribute(4).indexOfValue("positive");
		
		Instance inst2 = new DenseInstance(1, values2);
		dataset.add(inst2);

		
		
		System.out.println(dataset.toString());
		
		
		
		
		//Adds a new numeric attribute to the dataset in the second last position, to keep the label in the last one
		dataset.insertAttributeAt(new Attribute("1NewNumeric"), dataset.numAttributes()-1);
		
		
		System.out.println(dataset.toString());
		
		ReplaceMissingValues rp=new ReplaceMissingValues();
	
		
		try {
			rp.setInputFormat(dataset);
			dataset=Filter.useFilter(dataset, rp);
		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println(dataset.toString());

	}
	
	

}
