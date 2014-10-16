package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Twokenize;
import cmu.arktweetnlp.Tagger.TaggedToken;
import tsa.core.EmotionEvaluator;
import tsa.core.ExpandLexEvaluator;
import tsa.core.LexiconEvaluator;
import tsa.core.SWN3LexiconEvaluator;
import tsa.core.MyUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class LexExpandExpFilter extends SimpleBatchFilter {

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

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("LEX-CLPW")); // MetaLex Positive words
		att.add(new Attribute("LEX-CLNW")); // MetaLex Negative words
		
		att.add(new Attribute("LEX-EDPS")); // Edinburgh Pos Score
		att.add(new Attribute("LEX-EDNS")); // EdinBurgh Negative Score
		
		
		att.add(new Attribute("LEX-EDPW")); // Edinburgh Pos Words
		att.add(new Attribute("LEX-EDNW")); // EdinBurgh Negative Words
		
		
		att.add(new Attribute("LEX-S140PS")); // s140 Pos Score
		att.add(new Attribute("LEX-S140NS")); // s140 Negative Score
		
		
		att.add(new Attribute("LEX-S140PW")); // s140 Pos Words
		att.add(new Attribute("LEX-S140NW")); // s140 Negative Words
		
				
		att.add(new Attribute("LEX-CONPS")); // Consensus Pos Score
		att.add(new Attribute("LEX-CONNS")); // Consensus Negative Score
		
		att.add(new Attribute("LEX-CONPW")); // Consensus Pos Words
		att.add(new Attribute("LEX-CONNW")); // Consensus Negative Words
		
		
		

		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		// Instances result = new Instances(determineOutputFormat(instances),
		// 0);

		Instances result = getOutputFormat();

		// reference to the content of the tweet
		Attribute attrCont = instances.attribute("content");

		LexiconEvaluator cleanLex = new LexiconEvaluator(
				"lexicons/cleanLex.csv");
		cleanLex.processDict();
		
		ExpandLexEvaluator consLex=new ExpandLexEvaluator("lexicons/ConsLex.csv");
		consLex.processDict();
		
		ExpandLexEvaluator edLex=new ExpandLexEvaluator("lexicons/EdLex.csv");
		edLex.processDict();
		
		ExpandLexEvaluator s140Lex=new ExpandLexEvaluator("lexicons/s140Lex.csv");
		s140Lex.processDict();
		
		
		Tagger tagger = new Tagger();
		tagger.loadModel("models/model.20120919");
		
		
		
		
	
		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			content=content.toLowerCase();
			
			List<String> words =Twokenize.tokenizeRawTweetText(content);

			Map<String, Integer> cleanLexCounts = cleanLex
					.evaluatePolarityLexicon(words);
			values[result.attribute("LEX-CLPW").index()] = cleanLexCounts
					.get("posCount");
			values[result.attribute("LEX-CLNW").index()] = cleanLexCounts
					.get("negCount");

			
			
			
			List<String> tagWords=new ArrayList<String>();

			List<TaggedToken> tagTokens=tagger.tokenizeAndTag(content.toLowerCase());
			for(TaggedToken tt:tagTokens){
				tagWords.add(tt.tag+"-"+tt.token);
			}
		
		
			Map<String,Double> consScores=consLex.evaluatePolarity(tagWords);
			
			values[result.attribute("LEX-CONPS").index()] = consScores.get("posScore");
			values[result.attribute("LEX-CONNS").index()] = consScores.get("negScore");
			
			values[result.attribute("LEX-CONPW").index()] = consScores.get("posCount");
			values[result.attribute("LEX-CONNW").index()] = consScores.get("negCount");
			
			Map<String,Double> edScores=edLex.evaluatePolarity(tagWords);
			
			values[result.attribute("LEX-EDPS").index()] = edScores.get("posScore");
			values[result.attribute("LEX-EDNS").index()] = edScores.get("negScore");
			
			values[result.attribute("LEX-EDPW").index()] = edScores.get("posCount");
			values[result.attribute("LEX-EDNW").index()] = edScores.get("negCount");
			
			Map<String,Double> s140Scores=s140Lex.evaluatePolarity(tagWords);
			
			values[result.attribute("LEX-S140PS").index()] = s140Scores.get("posScore");
			values[result.attribute("LEX-S140NS").index()] = s140Scores.get("negScore");
			
			values[result.attribute("LEX-S140PW").index()] = s140Scores.get("posCount");
			values[result.attribute("LEX-S140NW").index()] = s140Scores.get("negCount");
			

			Instance inst = new SparseInstance(1, values);

			inst.setDataset(result);

			// copy possible strings, relational values...
			copyValues(inst, false, instances, result);

			result.add(inst);

		}

		return result;
	}

}
