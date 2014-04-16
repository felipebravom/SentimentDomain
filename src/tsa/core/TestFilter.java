package tsa.core;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

public class TestFilter {

	static public void main(String args[]) throws Exception{
		
		TweetCollectionToArff ta=new SemEvalToArff();
		Instances train=ta.createDataset("datasets/example.txt");
	//	System.out.println(dataset.toString());
		
	
		SimpleBatchFilter wordFilter=new CopyOfTwitterNlpWordToVector();
		
		
		
		//SimpleBatchFilter wordFilter=new SimpleBatch();
		wordFilter.setInputFormat(train);	
		
					
     	train=Filter.useFilter(train, wordFilter);
//		
//		wordFilter=new TwitterNlpPos();
//		wordFilter.setInputFormat(train);
//		
//		train= Filter.useFilter(train,wordFilter);
//		
//		wordFilter = new LexiconFilter();
//		wordFilter.setInputFormat(train);
//		
//		train= Filter.useFilter(train, wordFilter);
//		
		
	//	System.out.println(train);
		
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File("example_sent_data.arff"));
		saver.writeBatch();
		
		
	
//		Instances test=ta.createDataset("datasets/b.txt");
//		Instances test2=Filter.useFilter(test, wordFilter);
//		System.out.println(test2);
		

		/*

		SimpleBatchFilter posFilter=new TwitterNlpPos();
		posFilter.setInputFormat(wordDataset);
		Instances wordPosDataset=Filter.useFilter(wordDataset, posFilter);
		System.out.println(wordPosDataset);
		
		*/
		
		
		
		//Filter.
		
	}
	
	
}
