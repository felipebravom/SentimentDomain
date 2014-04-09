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
		return "A simple batch filter that adds attributes for all the "
				+ "Twitter-oriented POS tags of the TwitterNLP library.  ";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		att.add(new Attribute("LEX-OFPW")); // OpinionFinder Positive words
		att.add(new Attribute("LEX-OFNW")); // OpinionFinder Negative words
		att.add(new Attribute("LEX-BLPW")); // Bing Liu Positive words
		att.add(new Attribute("LEX-BLNW")); // Bing Liu Negative words


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
		
		LexiconEvaluator opFinderLex = new LexiconEvaluator("lexicons/opinion-finder.txt");
		opFinderLex.processDict();
		LexiconEvaluator bingLiuLex = new LexiconEvaluator("lexicons/BingLiu.csv");
		bingLiuLex.processDict();
		

		for (int i = 0; i < instances.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			String content = instances.instance(i).stringValue(attrCont);
			List<String> words = Utils.cleanTokenize(content);
			
			Map<String,Integer> opFinderCounts=Utils.evaluatePolarityLexicon(words,opFinderLex);
			values[result.attribute("LEX-OFPW").index()]=opFinderCounts.get("posCount");
			values[result.attribute("LEX-OFNW").index()]=opFinderCounts.get("negCount");
			
			Map<String,Integer> bingLiuCounts=Utils.evaluatePolarityLexicon(words,opFinderLex);
			values[result.attribute("LEX-BLPW").index()]=bingLiuCounts.get("posCount");
			values[result.attribute("LEX-BLNW").index()]=bingLiuCounts.get("negCount");


			
			
			Instance inst = new SparseInstance(1, values);
			result.add(inst);
			
		}
		
		
		return result;
	}

}
