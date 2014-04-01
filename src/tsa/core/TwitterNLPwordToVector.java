package tsa.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import cmu.arktweetnlp.Twokenize;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class TwitterNLPwordToVector extends SimpleBatchFilter {
	
	private Map<String,Integer> vocDocFreq; // the vocabulary and the number of tweets in which the word appears
	private List<Map<String,Integer>> wordVecs; // List of word vectors with their corresponding frequencies per tweet

	
	
	
	@Override
	public String globalInfo() {
	     return   "A simple batch filter that adds an additional attribute 'bla' at the end "
	             + "containing the index of the processed instance.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat){
		this.vocDocFreq=new HashMap<String,Integer>();
		this.wordVecs=new ArrayList<Map<String,Integer>>();
		
		
		// reference to the content of the tweet
		Attribute attrCont=inputFormat.attribute("content");
				
		for(ListIterator<Instance> it=inputFormat.listIterator();it.hasNext();){
			Instance inst=it.next();
			String content=inst.stringValue(attrCont);
			
			List<String> tokens=this.cleanTokenize(content);
			Map<String,Integer> wordFreqs=calculateTermFreq(tokens);
			this.wordVecs.add(wordFreqs);
			
			for(String word:wordFreqs.keySet()){
				if(this.vocDocFreq.containsKey(word)){
					this.vocDocFreq.put(word, this.vocDocFreq.get(word)+1);
				}
								
			}
			

			
			
			
			
			System.out.println(content);
		
			
		}
		
		
	
	
		
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
	
	
	// Tokenize a the tweet using TwitterNLP and cleans repeated letters, URLS, and user mentions
	 public List<String> cleanTokenize(String content){
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

	

	   public static void main(String[] args) {
		     runFilter(new TwitterNLPwordToVector(), args);
		   }
	
	
}

