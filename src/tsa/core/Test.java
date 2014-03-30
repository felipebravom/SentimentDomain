package tsa.core;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Twokenize;
import cmu.arktweetnlp.Tagger.TaggedToken;
import cmu.arktweetnlp.impl.Model;
import cmu.arktweetnlp.impl.ModelSentence;
import cmu.arktweetnlp.impl.Sentence;
import cmu.arktweetnlp.impl.features.FeatureExtractor;
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
	
	public static List<TaggedToken> tokenizeAndTag(Tagger tagger,List<String> tokens) {
		Model model=tagger.model;
		FeatureExtractor featureExtractor=tagger.featureExtractor;

		Sentence sentence = new Sentence();
		sentence.tokens = tokens;
		ModelSentence ms = new ModelSentence(sentence.T());
		featureExtractor.computeFeatures(sentence, ms);
		model.greedyDecode(ms, false);

		ArrayList<TaggedToken> taggedTokens = new ArrayList<TaggedToken>();

		for (int t=0; t < sentence.T(); t++) {
			TaggedToken tt = new TaggedToken();
			tt.token = tokens.get(t);
			tt.tag = model.labelVocab.name( ms.labels[t] );
			taggedTokens.add(tt);
		}

		return taggedTokens;
	}
	
	
	// Returns POS tags from a List of tokens using TwitterNLP	
	public static List<String> getPOStags(Tagger tagger,List<String> tokens){

		Sentence sentence = new Sentence();
		sentence.tokens = tokens;
		ModelSentence ms = new ModelSentence(sentence.T());
		tagger.featureExtractor.computeFeatures(sentence, ms);
		tagger.model.greedyDecode(ms, false);

		ArrayList<String> tags = new ArrayList<String>();

		for (int t=0; t < sentence.T(); t++) {
			String tag = tagger.model.labelVocab.name( ms.labels[t] );
			tags.add(tag);
		}

		return tags;
		
	}
	
	
	static public List<String> cleanTokenize(String content){
		content=content.toLowerCase();
		//content=content.replaceAll("([aeiou])\\1+","$1"); // remove repeated vowels
		
		content=content.replaceAll("([a-z])\\1+","$1$1"); // if a letters appears two or more times is replaced by two occurrences
		
			
		List<String> tokens=new ArrayList<String>();

		for(String word:Twokenize.tokenizeRawTweetText(content)){

			String cleanWord=word; 

			// Replace URLs for a special token URL
			if(word.matches("http.*|ww\\..*")){
				cleanWord="http://www.url.com";
			}

			// Replace user mentions to a special token USER
			else if(word.matches("@.*")){
				cleanWord="@user";
			}	

			tokens.add(cleanWord);


		}

		return tokens;		

	}
	


	public static void main(String[] args) throws IOException {
		
	       Pattern pattern = Pattern.compile("([a-z])\\1{3}");
	        Matcher matcher = pattern.matcher("asdffffffasdf");
	        System.out.println(matcher.find());
		
		String content="Looooks I wwola ww.g.co loooooove youuuuu arrrrrghhh like Andy the Android may have had a little @user too much https://lal !!!!! www.google.cl fun yesterday. http://t.co/7ZDEfzEC";
		
		
		List<String> cleanTokens=cleanTokenize(content);
		for(String word:cleanTokens){
			System.out.println(word);
		}
		
		
		/*
		Tagger tagger=new Tagger();
		tagger.loadModel("models/model.20120919");
		Model m=tagger.model;
		FeatureExtractor f=tagger.featureExtractor;
		
		
		String tweet="ikr smh he asked fir yo last name so he can add u on fb lololol by accident obam";
		List<String> words= Twokenize.tokenizeRawTweetText(tweet);
		
		List<TaggedToken> tokens=tokenizeAndTag(tagger,words);
		for(TaggedToken token:tokens){
			System.out.println(token.token+" "+token.tag);
		}
		
		List<String> tags=getPOStags(tagger,words);

		
		for(int i=0;i<words.size();i++){
			System.out.println(words.get(i)+" "+tags.get(i));
		}
		
		
	
		
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
		
		*/

	}
	
	

}
