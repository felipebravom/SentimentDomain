package tsa.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Twokenize;
import cmu.arktweetnlp.impl.ModelSentence;
import cmu.arktweetnlp.impl.Sentence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingWithUserConstant;

public class CreateDatasetSimple {
	private String inputFile; // The file from which the data is read	
	private Tagger tagger; // The POS tagger
	private Instances dataset; // the Weka datasaet
	
	public CreateDatasetSimple(String inputFile){
		this.inputFile=inputFile;
	}
	
	
	public Instances getDataset(){
		return this.dataset;
	}
	
	public void setupInstances(){
		ArrayList<Attribute> attributes=new ArrayList<Attribute>();
		
		// Add POS features
		attributes.add(new Attribute("POS-N")); // common noun
		attributes.add(new Attribute("POS-O")); // personal or WH, not possessive
		attributes.add(new Attribute("POS-S")); // nominal + possessive
		attributes.add(new Attribute("POS-^")); // proper noun
		attributes.add(new Attribute("POS-Z")); // proper noun + possessive  
		attributes.add(new Attribute("POS-L")); // nominal + verbal
		attributes.add(new Attribute("POS-M")); // proper noun + verbal
		attributes.add(new Attribute("POS-V")); // verb or auxiliary
		attributes.add(new Attribute("POS-A")); // adjective
		attributes.add(new Attribute("POS-R")); // adverb
		attributes.add(new Attribute("POS-!")); // interjection
		attributes.add(new Attribute("POS-D")); // determiner
		attributes.add(new Attribute("POS-P")); // preposition or subordinating conjunction
		attributes.add(new Attribute("POS-&")); // coordinating conjunction
		attributes.add(new Attribute("POS-T")); // verb particle
		attributes.add(new Attribute("POS-X")); // existential "there" or predeterminer
		attributes.add(new Attribute("POS-Y")); // X + verbal
		attributes.add(new Attribute("POS-#")); // hashtag
		attributes.add(new Attribute("POS-@")); // at-mention
		attributes.add(new Attribute("POS-~")); // Twitter discourse function word
		attributes.add(new Attribute("POS-U")); // URL or email address
		attributes.add(new Attribute("POS-E")); // emoticon
		attributes.add(new Attribute("POS-$")); // numeral
		attributes.add(new Attribute("POS-,")); // punctuation
		attributes.add(new Attribute("POS-G")); // other abbreviation, foreign word, possessive ending, symbol, or garbage
		attributes.add(new Attribute("POS-?")); // unsure
		
		
		// Create the  label
		ArrayList<String> label=new ArrayList<String>();
		label.add("0"); // for sparse notation
		label.add("positive");
		label.add("neutral");
		label.add("negative");
		
		attributes.add(new Attribute("CLASS",label));	
		
		this.dataset=new Instances("Twitter Sentiment Analysis Dataset", attributes,0); // The last attribute 		
	
	}
	
	// setups the Tagger with the corresponding model
	public boolean setupTagger(String model){
		this.tagger=new Tagger();
		try {
			this.tagger.loadModel(model);
			return true;
		} catch (IOException e) {
			return false;
		}
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
	
	
	// Tokenize a the tweet using TwitterNLP and cleans repeated letters, URLS, and user mentions
	static public List<String> cleanTokenize(String content){
		content=content.toLowerCase();		
		
		// if a letters appears two or more times it is replaced by only two occurrences of it
		content=content.replaceAll("([a-z])\\1+","$1$1"); 
					
		List<String> tokens=new ArrayList<String>();

		for(String word:Twokenize.tokenizeRawTweetText(content)){
			String cleanWord=word; 

			// Replace URLs to a generic URL
			if(word.matches("http.*|ww\\..*")){
				cleanWord="http://www.url.com";
			}

			// Replaces user mentions to a generic user
			else if(word.matches("@.*")){
				cleanWord="@user";
			}	

			tokens.add(cleanWord);
		}
		return tokens;		
	}
	
	
	// calculates the frequency of each different token in a list of strings
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
	
	
	// Returns POS tags from a List of tokens using TwitterNLP	
	public List<String> getPOStags(List<String> tokens){

		Sentence sentence = new Sentence();
		sentence.tokens = tokens;
		ModelSentence ms = new ModelSentence(sentence.T());
		this.tagger.featureExtractor.computeFeatures(sentence, ms);
		this.tagger.model.greedyDecode(ms, false);

		ArrayList<String> tags = new ArrayList<String>();

		for (int t=0; t < sentence.T(); t++) {
			String tag = "POS-"+tagger.model.labelVocab.name( ms.labels[t] );
			tags.add(tag);
		}

		return tags;
		
	}
	
	
	
	
	
	public void processData(){
		try {
			BufferedReader bf = new BufferedReader(new FileReader(this.inputFile));
			String line;			
			while( (line=bf.readLine())!=null){
				String parts[]=line.split("\t");
				
				String label=parts[2];
				String content=parts[3];
				
				// extract tokens
				List<String> tokens= cleanTokenize(content);
				
				// calculate frequencies of different tokens
				Map<String,Integer> wordFreqs=calculateTermFreq(tokens);
												
				
				// Add new attributes for each new word
				for(String word:wordFreqs.keySet()){
					if(!this.checkAttribute(word))
						this.addNumericAttribute(word);
				}
				
				
				// get POS tags for all the different tokens
				List<String> posTags= this.getPOStags(tokens);
				
				// calculate frequencies of different POS tags 
				Map<String,Integer> posFreqs=calculateTermFreq(posTags);
				
				
				
				// Add data with the frequencies
				int numAtt=this.dataset.numAttributes();
				double values[] = new double[numAtt];
				
				// add word values
				for(String word:wordFreqs.keySet()){
					int index=this.dataset.attribute(word).index();
					values[index]=wordFreqs.get(word);					
				}
				
				// add POS values
				for(String posTag:posFreqs.keySet()){
					int index=this.dataset.attribute(posTag).index();
					values[index]=posFreqs.get(posTag);					
				}
				
				
				
				// add class label
				if(label.equals("positive"))
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("positive");
				else if(label.equals("neutral")||label.equals("objective")||label.equals("objective-OR-neutral"))
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("neutral");
				else
					values[numAtt-1]=this.dataset.attribute(numAtt-1).indexOfValue("negative");
				
				
				Instance inst = new SparseInstance(1, values);
				this.dataset.add(inst);				
				
								
		
			}
			
						
			bf.close();
			
			
			// Replace missing values caused by adding new attributes in each iteration
			ReplaceMissingWithUserConstant rp=new ReplaceMissingWithUserConstant();
			rp.setNumericReplacementValue("0");
			rp.setInputFormat(this.dataset);
			this.dataset=Filter.useFilter(this.dataset, rp);
			
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	public boolean exportInstances(String fileName){
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(this.dataset);
		try {
			saver.setFile(new File(fileName));
			saver.writeBatch();
			return true;
		} catch (IOException e) {
			return false;
		}
		
	}
	
	
	static public void main(String args[]){
		CreateDatasetSimple cd=new CreateDatasetSimple(args[0]);
		cd.setupInstances();
		cd.setupTagger("models/model.20120919");

		cd.processData();
		cd.exportInstances(args[1]);
		
		
	}

}
