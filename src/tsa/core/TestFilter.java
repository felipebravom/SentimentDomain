package tsa.core;

import java.io.File;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

public class TestFilter {

	static public void main(String args[]) throws Exception{
		
		TweetCollectionToArff ta=new SemEvalToArff();
		Instances dataset=ta.createDataset("datasets/twitter-train-B.txt");
	//	System.out.println(dataset.toString());
		
		SimpleBatchFilter wordFilter=new TwitterNLPwordToVector();
		wordFilter.setInputFormat(dataset);	
		
		
		
		Filter.useFilter(dataset, wordFilter);
		
		//Filter.
		
	}
	
	
}
