package tsa.core;

import java.io.IOException;
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
import weka.filters.Filter;
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


		String content="Ron Paul Snr Advisor Doug Wead Interview with Frost &#8211; Mar 31 2012 http://t.co/tdzhFWYN";
		Tagger tagger = new Tagger();
		tagger.loadModel("models/model.20120919");
		
		
		List<String> tagWords=new ArrayList<String>();

		List<TaggedToken> tagTokens=tagger.tokenizeAndTag(content.toLowerCase());
		for(TaggedToken tt:tagTokens){
			tagWords.add(tt.tag+"-"+tt.token);
		}
	
		for(String pal:tagWords){
			System.out.println(pal);
		}
		

	}



}
