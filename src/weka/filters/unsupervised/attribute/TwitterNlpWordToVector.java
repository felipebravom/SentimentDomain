package weka.filters.unsupervised.attribute;


import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.TreeMap;

import tsa.core.MyUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

public class TwitterNlpWordToVector extends SimpleBatchFilter {

	/**
	 * 
	 */


	private static final long serialVersionUID = 3635946466523698211L;

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

	// To allow determineOutputFormat to access to entire dataset
	public boolean allowAccessToFullInputFormat() {
		return true;
	}

	// Calculates the vocabulary and the word vectors from an Instances object
	// The vocabulary is only extracted the first time the filter is run.
	public void computeWordVecsAndVoc(Instances inputFormat) {

		// The vocabulary is created only in the first execution
		if (!this.isFirstBatchDone())
			this.vocDocFreq = new TreeMap<String, Integer>();

		this.wordVecs = new ArrayList<Map<String, Integer>>();

		// reference to the content of the tweet
		Attribute attrCont = inputFormat.attribute("content");

		for (ListIterator<Instance> it = inputFormat.listIterator(); it
				.hasNext();) {
			Instance inst = it.next();
			String content = inst.stringValue(attrCont);

			// tokenizes the content
			List<String> tokens = MyUtils.cleanTokenize(content);
			Map<String, Integer> wordFreqs = MyUtils.calculateTermFreq(tokens);

			// Add the frequencies of the different words
			this.wordVecs.add(wordFreqs);

			// The vocabulary is calculated only the first time we run the
			// filter
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

	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) {

		ArrayList<Attribute> att = new ArrayList<Attribute>();

		// Adds all attributes of the inputformat
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			att.add(inputFormat.attribute(i));
		}

		// calculates the word frequency vectors and the vocabulary
		this.computeWordVecsAndVoc(inputFormat);

		for (String word : this.vocDocFreq.keySet()) {

			Attribute a = new Attribute("WORD-" + word);

			att.add(a); // adds an attribute for each word using a prefix

			// pw.println("word: " + word + " bytes: "
			// + Arrays.toString(word.getBytes()) + " attribute name: "
			// + a.name() + " HashValue:" + this.vocDocFreq.get(word));

		}

		Instances result = new Instances("Twitter Sentiment Analysis", att, 0);

		// set the class index
		result.setClassIndex(inputFormat.classIndex());

		return result;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {

		Instances result = getOutputFormat();

		// if we are in the testing data we calculate the word vectors again
		if (this.isFirstBatchDone()) {
			this.computeWordVecsAndVoc(instances);
		}

		// System.out.println("++++" + instances);

		int i = 0;
		for (Map<String, Integer> wordVec : this.wordVecs) {
			double[] values = new double[result.numAttributes()];

			// copy previous attributes values
			for (int n = 0; n < instances.numAttributes(); n++)
				values[n] = instances.instance(i).value(n);

			// add words using the frequency as attribute value
			for (String word : wordVec.keySet()) {
				// we only add the value if the word was previously included
				// into the vocabulary, otherwise we discard it
				if (result.attribute("WORD-" + word) != null)
					values[result.attribute("WORD-" + word).index()] = wordVec
							.get(word);

			}

			Instance inst = new SparseInstance(1, values);

			inst.setDataset(result);
			// copy possible strings, relational values...
			copyValues(inst, false, instances, result);

			result.add(inst);
			i++;

		}

		return result;
	}

	public static void main(String[] args) {
		runFilter(new TwitterNlpWordToVector(), args);
	}

}
