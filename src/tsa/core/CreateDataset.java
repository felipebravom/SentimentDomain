package tsa.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Twokenize;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingWithUserConstant;

public class CreateDataset {
	private String inputFile; // The file from which the data is read	
	private Instances dataset; // the Weka datasaet
	
	public CreateDataset(String inputFile){
		this.inputFile=inputFile;
	}
	
	
	public Instances getDataset(){
		return this.dataset;
	}
	
	public void setupInstances(){
		ArrayList<Attribute> attributes=new ArrayList<Attribute>();
		
		// Create the 
		ArrayList<String> label=new ArrayList<String>();
		label.add("positive");
		label.add("neutral");
		label.add("negative");
		
		attributes.add(new Attribute("CLASS",label));	
		
		this.dataset=new Instances("Twitter Sentiment Analysis Dataset", attributes,0); // The last attribute 		
	
	}
	
	// Checks whether a certain attribute is part of the dataset
	public boolean checkAttribute(String attr){
		if((this.dataset.attribute(attr)!=null))
				return true;
		else
			return false;		
	}
	
	
	
	//Adds a new numeric attribute to the dataset in the second last position. The label is always keep on the last one
	// Its important to check before whether the attribute has been added before
	public void addNumericAttribute(String attr){
		this.dataset.insertAttributeAt(new Attribute(attr), dataset.numAttributes()-1);
	}
	
	
	// calculates the frequency of each different token in a list of words
	public Map<String,Integer> calculateTermFreq(List<String> tokens){
		Map<String,Integer> termFreq=new HashMap<String,Integer>();
		
		
		// Traverse the strings and increments the counter when the token was already seen before
		for(String token:tokens){
			if(termFreq.containsKey(token))
				termFreq.put(token, termFreq.get(token)+1);
			else
				termFreq.put(token, 1);			
		}
			
		return termFreq;		
	}
	
	
	
	
	
	
	public void processData(){
		try {
			BufferedReader bf = new BufferedReader(new FileReader(this.inputFile));
			String line;			
			while( (line=bf.readLine())!=null){
				String parts[]=line.split("\t");
				System.out.println(parts.length);
				System.out.println(parts[3]);
				System.out.println(parts[2]);
				
				String label=parts[2];
				String content=parts[3];
				
				List<String> tokens= Twokenize.tokenizeRawTweetText(content.toLowerCase());
				Map<String,Integer> freqs=calculateTermFreq(tokens);
			
				
				// Add new attributes for each unseen word
				for(String word:freqs.keySet()){
					if(!this.checkAttribute(word))
						this.addNumericAttribute(word);
				}
				
				// Add data with the frequencies
				int numAtt=this.dataset.numAttributes();
				double values[] = new double[numAtt];
				
				for(String word:freqs.keySet()){
					int index=this.dataset.attribute(word).index();
					values[index]=freqs.get(word);					
				}
				
				// add class label
				if(label.equals("positive"))
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("positive");
				else if(label.equals("neutral")||label.equals("objective")||label.equals("objective-OR-neutral"))
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("neutral");
				else
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("negative");
				
				
				Instance inst = new DenseInstance(1, values);
				this.dataset.add(inst);				
				
								
		
			}
			
						
			bf.close();
			
			
			// Replace missing values caused by adding new attributes in each iteration
			ReplaceMissingWithUserConstant rp=new ReplaceMissingWithUserConstant();
			rp.setNumericReplacementValue("0");
			System.out.println(rp.getNominalStringReplacementValue());
			rp.setInputFormat(this.dataset);
			this.dataset=Filter.useFilter(this.dataset, rp);
			
			
			
			
		
		
			ArffSaver saver = new ArffSaver();
			saver.setInstances(this.dataset);
			saver.setFile(new File("test.arff"));
			saver.writeBatch();
		
		
		
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	static public void main(String args[]){
		CreateDataset cd=new CreateDataset("datasets/example.txt");
		cd.setupInstances();

		
		System.out.println(cd.getDataset());
		
		cd.processData();
		
		System.out.println(cd.getDataset());
		
	}

}
