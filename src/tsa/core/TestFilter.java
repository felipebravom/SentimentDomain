package tsa.core;

import java.io.File;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

public class TestFilter {

	static public void main(String args[]) throws Exception{
		
		TweetCollectionToArff ta=new SemEvalToArff();
		Instances train=ta.createDataset("datasets/a.txt");
	//	System.out.println(dataset.toString());
		
	
		SimpleBatchFilter wordFilter=new TwitterNlpWordToVector();
		
		
		
		//SimpleBatchFilter wordFilter=new SimpleBatch();
		wordFilter.setInputFormat(train);	
		
					
		Instances wordDataset=Filter.useFilter(train, wordFilter);
		System.out.println(wordDataset);
	
		Instances test=ta.createDataset("datasets/b.txt");
		Instances test2=Filter.useFilter(test, wordFilter);
		System.out.println(test2);
		

		/*

		SimpleBatchFilter posFilter=new TwitterNlpPos();
		posFilter.setInputFormat(wordDataset);
		Instances wordPosDataset=Filter.useFilter(wordDataset, posFilter);
		System.out.println(wordPosDataset);
		
		*/
		
		
		
		//Filter.
		
	}
	
	
}
