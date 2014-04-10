package tsa.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Tagger;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class LexiconFilter extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4983739424598292130L;

	@Override
	public String globalInfo() {
		return "A batch filter that calcuates attributes from different lexical resources for Sentiment Analysis ";

	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		
		if(!this.isFirstBatchDone()){

		System.out.println("PASADA");
		}
		
		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("LEX-OFPW")); // OpinionFinder Positive words
		att.add(new Attribute("LEX-OFNW")); // OpinionFinder Negative words
		att.add(new Attribute("LEX-BLPW")); // Bing Liu Positive words
		att.add(new Attribute("LEX-BLNW")); // Bing Liu Negative words
		att.add(new Attribute("LEX-AFPS")); // AFINN Positive score
		att.add(new Attribute("LEX-AFNS")); // AFINN Negative score
		att.add(new Attribute("LEX-S140PS")); // Sentiment140 Positive score
		att.add(new Attribute("LEX-S140NS")); // Sentiment140 Negative score
		att.add(new Attribute("LEX-NRCHASHPS")); // NRCHashtag Positive score
		att.add(new Attribute("LEX-NRCHASHNS")); // NRCHastag Negative score
		att.add(new Attribute("LEX-SWN3PS")); // SentiWordnet Positive score
		att.add(new Attribute("LEX-SWN3NS")); // SentiWordnet Negative score
		
		att.add(new Attribute("LEX-NRC-emo-anger")); // NRC-emotion anger words
		att.add(new Attribute("LEX-NRC-emo-anticipation")); // NRC-emotion anticipation words
		att.add(new Attribute("LEX-NRC-emo-disgust")); // NRC-disgust disgust words
		att.add(new Attribute("LEX-NRC-emo-fear")); // NRC-emotion fear words
		att.add(new Attribute("LEX-NRC-emo-joy")); // NRC-emotion joy words
		att.add(new Attribute("LEX-NRC-emo-sadness")); // NRC-emotion sadness words
		att.add(new Attribute("LEX-NRC-emo-surprise")); // NRC-emotion surprise words
		att.add(new Attribute("LEX-NRC-emo-trust")); // NRC-emotion trust emotion words
		
		att.add(new Attribute("LEX-NRC-emo-PW")); // NRC-emotion positive words
		att.add(new Attribute("LEX-NRC-emo-NW")); // NRC-emotion negative words
		
		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(determineOutputFormat(instances), 0);
				
		// reference to the content of the tweet
		Attribute attrCont = instances.attribute("content");

		LexiconEvaluator opFinderLex = new LexiconEvaluator(
				"lexicons/opinion-finder.txt");
		opFinderLex.processDict();
		LexiconEvaluator bingLiuLex = new LexiconEvaluator(
				"lexicons/BingLiu.csv");
		bingLiuLex.processDict();

		LexiconEvaluator afinnLex = new LexiconEvaluator(
				"lexicons/AFINN-111.txt");
		afinnLex.processDict();

		LexiconEvaluator s140Lex = new LexiconEvaluator(
				"lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt");
		s140Lex.processDict();

		LexiconEvaluator nrcHashLex = new LexiconEvaluator(
				"lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt");
		nrcHashLex.processDict();

		LexiconEvaluator swn3Lex = new SWN3LexiconEvaluator(
				"lexicons/SentiWordNet_3.0.0.txt");
		swn3Lex.processDict();
		
		EmotionEvaluator nrcEmoLex = new EmotionEvaluator(
				"lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt");
		nrcEmoLex.processDict();
		

		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			List<String> words = Utils.cleanTokenize(content);

			Map<String, Integer> opFinderCounts = opFinderLex
					.evaluatePolarityLexicon(words);
			values[result.attribute("LEX-OFPW").index()] = opFinderCounts
					.get("posCount");
			values[result.attribute("LEX-OFNW").index()] = opFinderCounts
					.get("negCount");

			Map<String, Integer> bingLiuCounts = bingLiuLex
					.evaluatePolarityLexicon(words);
			values[result.attribute("LEX-BLPW").index()] = bingLiuCounts
					.get("posCount");
			values[result.attribute("LEX-BLNW").index()] = bingLiuCounts
					.get("negCount");

			Map<String, Double> afinnCounts = afinnLex
					.evaluateStrengthLexicon(words);
			values[result.attribute("LEX-AFPS").index()] = afinnCounts
					.get("posScore");
			values[result.attribute("LEX-AFNS").index()] = afinnCounts
					.get("negScore");

			Map<String, Double> s140LexCounts = s140Lex
					.evaluateStrengthLexicon(words);
			values[result.attribute("LEX-S140PS").index()] = s140LexCounts
					.get("posScore");
			values[result.attribute("LEX-S140NS").index()] = s140LexCounts
					.get("negScore");

			Map<String, Double> nrcHashCounts = nrcHashLex
					.evaluateStrengthLexicon(words);
			values[result.attribute("LEX-NRCHASHPS").index()] = nrcHashCounts
					.get("posScore");
			values[result.attribute("LEX-NRCHASHNS").index()] = nrcHashCounts
					.get("negScore");

			Map<String, Double> swn3Counts = swn3Lex
					.evaluateStrengthLexicon(words);
			values[result.attribute("LEX-SWN3PS").index()] = swn3Counts
					.get("posScore");
			values[result.attribute("LEX-SWN3NS").index()] = swn3Counts
					.get("negScore");
			
			
			
			Map<String,Integer> nrcEmoCounts = nrcEmoLex.evaluateEmotion(words);
			values[result.attribute("LEX-NRC-emo-anger").index()] = nrcEmoCounts
					.get("anger");
			values[result.attribute("LEX-NRC-emo-anticipation").index()] = nrcEmoCounts
					.get("anticipation");
			values[result.attribute("LEX-NRC-emo-disgust").index()] = nrcEmoCounts
					.get("disgust");
			values[result.attribute("LEX-NRC-emo-fear").index()] = nrcEmoCounts
					.get("fear");
			values[result.attribute("LEX-NRC-emo-joy").index()] = nrcEmoCounts
					.get("joy");
			values[result.attribute("LEX-NRC-emo-sadness").index()] = nrcEmoCounts
					.get("sadness");
			values[result.attribute("LEX-NRC-emo-surprise").index()] = nrcEmoCounts
					.get("surprise");
			values[result.attribute("LEX-NRC-emo-trust").index()] = nrcEmoCounts
					.get("trust");
			
			values[result.attribute("LEX-NRC-emo-PW").index()] = nrcEmoCounts
					.get("positive");
			values[result.attribute("LEX-NRC-emo-NW").index()] = nrcEmoCounts
					.get("negative");

			
		
			
			Instance inst = new SparseInstance(1, values);
			result.add(inst);

		}

		return result;
	}

}
