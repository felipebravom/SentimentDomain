package tsa.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.HumanCodedPolarityToArff;
import weka.core.converters.HumanCodedToArff;
import weka.core.converters.SandersToArff;
import weka.core.converters.SemEvalToArff;
import weka.core.converters.TweetCollectionToArff;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.LexiconFilter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.TwitterNlpPos;
import weka.filters.unsupervised.attribute.TwitterNlpWordToVector;

public class TestFilter {

	static public void main(String args[]) throws Exception{

		//		TweetCollectionToArff ta=new HumanCodedPolarityToArff();
		//		Instances train=ta.createDataset("datasets/6humancoded/twitter4242.txt");

		BufferedReader reader = new BufferedReader(
				new FileReader("/some/where/data.arff"));
		Instances train = new Instances(reader);
		reader.close();
		// setting class attribute
		train.setClassIndex(train.numAttributes() - 1);

		
		long startTime=System.nanoTime();
		
		SimpleBatchFilter f=new TwitterNlpWordToVector();
		f.setOptions(Utils.splitOptions(" -I 0 -P LALA-"));

		f.setInputFormat(train);


		train=Filter.useFilter(train, f);
		
		System.out.println(System.nanoTime()-startTime);


		//	System.out.println(dataset.toString());

		//		MultiFilter multFilt=new MultiFilter();
		//	//	multFilt.
		//		
		//		
		//		List<Filter> filters=new ArrayList<Filter>();
		//		filters.add(new TwitterNlpWordToVector());
		//		filters.add(new TwitterNlpPos());
		//		filters.add(new LexiconFilter());
		//
		//		
		//		// Discards the content and moves the class value to the end
		////		Reorder reorder=new Reorder();
		////		reorder.setOptions(Utils.splitOptions("-R 3-last,2"));		
		////		filters.add(reorder);
		//		
		//		multFilt.setFilters(filters.toArray(new Filter[0]));
		//		
		//		multFilt.setInputFormat(train);
		//		
		//		
		//		train=Filter.useFilter(train, multFilt);




		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File("TESTING.arff"));
		saver.writeBatch();



		//		SimpleBatchFilter wordFilter=new CopyOfTwitterNlpWordToVector();
		//			
		//		wordFilter.setInputFormat(train);	
		//						
		//     	train=Filter.useFilter(train, wordFilter);

		//		wordFilter=new TwitterNlpPos();
		//		wordFilter.setInputFormat(train);
		//		
		//		train= Filter.useFilter(train,wordFilter);
		//		
		//		wordFilter = new LexiconFilter();
		//		wordFilter.setInputFormat(train);
		//		
		//		train= Filter.useFilter(train, wordFilter);



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
