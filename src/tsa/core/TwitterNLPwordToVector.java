package tsa.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class TwitterNlpWordToVector extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private Map<String, Integer> vocDocFreq; // the vocabulary and the number of
												// tweets where the word appears
	private List<Map<String, Integer>> wordVecs; // List of word vectors with
													// their corresponding
													// frequencies per tweet

	@Override
	public String globalInfo() {
		return "A simple batch filter that adds attributes for all the "
				+ "Twitter-oriented POS tags of the TwitterNLP library.  ";
	}

	public boolean allowAccessToFullInputFormat() {
		return true;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {

		// The vocabulary is created only at the first time
		if (!this.isFirstBatchDone())
			this.vocDocFreq = new HashMap<String, Integer>();

		this.wordVecs = new ArrayList<Map<String, Integer>>();

		// reference to the content of the tweet
		Attribute attrCont = inputFormat.attribute("content");

		for (ListIterator<Instance> it = inputFormat.listIterator(); it
				.hasNext();) {
			Instance inst = it.next();
			String content = inst.stringValue(attrCont);

			// tokenizes the content
			List<String> tokens = Utils.cleanTokenize(content);
			Map<String, Integer> wordFreqs = Utils.calculateTermFreq(tokens);

			// Add the frequencies of the different words
			this.wordVecs.add(wordFreqs);

			// The vocabulary is calculated from the training test
			if (!this.isFirstBatchDone()) {
				// if the word is new we add it to the vocabulary, otherwise we
				// increment the document count
				for (String word : wordFreqs.keySet()) {
					if (this.vocDocFreq.containsKey(word)) {
						this.vocDocFreq
								.put(word, this.vocDocFreq.get(word) + 1);
					} else
						this.vocDocFreq.put(word, 1);

				}

			}

		}

		System.out.println("IS FIRST BATCH" + this.isFirstBatchDone());

		// sorts the words alphabetically
		String[] wordsArray = this.vocDocFreq.keySet().toArray(new String[0]);
		Arrays.sort(wordsArray);

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		for (String word : wordsArray) {
			att.add(new Attribute("WORD-" + word)); // adds an attribute for
													// each word using a prefix
		}

		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(determineOutputFormat(instances), 0);

		int i = 0;
		for (Map<String, Integer> wordVec : this.wordVecs) {
			double[] values = new double[result.numAttributes()];

			// copies previous attributes values
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			// adds the words using the frequency as attribute value
			for (String word : wordVec.keySet()) {
				// we only add the value if the word was previously included
				// into the vocabulary
				if (result.attribute("WORD-" + word) != null)
					values[result.attribute("WORD-" + word).index()] = wordVec
							.get(word);

			}

			Instance inst = new SparseInstance(1, values);
			result.add(inst);
			i++;

		}

		return result;
	}

	public static void main(String[] args) {
		runFilter(new TwitterNlpWordToVector(), args);
	}

}
