package tsa.core;

import java.io.File;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

public class TestFilter {

	static public void main(String args[]) throws Exception{
		
		TweetCollectionToArff ta=new SemEvalToArff();
		Instances dataset=ta.createDataset("datasets/example.txt");
	//	System.out.println(dataset.toString());
		
		SimpleBatchFilter wordFilter=new TwitterNLPwordToVector();
		wordFilter.setInputFormat(dataset);	
		
		
		
		
		
		Instances example=Filter.useFilter(dataset, wordFilter);
		System.out.println(example);
		
		//Filter.
		
	}
	
	
}
