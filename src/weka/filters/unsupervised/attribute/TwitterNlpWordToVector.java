package weka.filters.unsupervised.attribute;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import tsa.core.MyUtils;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

public class TwitterNlpWordToVector extends SimpleBatchFilter {

	/**  Converts one String attribute into a set of attributes
	 * representing word occurrence based on the TwitterNLP tokenizer.
	 * 
	 */


	/** for serialization */
	private static final long serialVersionUID = 3635946466523698211L;

	/** the vocabulary and the number of documents where the word appears */
	protected Map<String, Integer> vocDocFreq; 

	/** List of word vectors with their corresponding frequencies per tweet */
	protected List<Map<String, Integer>> wordVecs; 

	/** the index of the string attribute to be processed */
	protected int textIndex=0; 



	/** the index of the string attribute to be processed */
	protected String prefix="WORD-";

	@Override
	public String globalInfo() {
		return "A simple batch filter that adds attributes for all the "
				+ "Twitter-oriented POS tags of the TwitterNLP library.  ";
	}



	@Override
	public Capabilities getCapabilities() {
		
		Capabilities result = new Capabilities(this);
		result.disableAll();
		
		
		
		 // attributes
	    result.enableAllAttributes();
	    result.enable(Capability.MISSING_VALUES);

	    // class
	    result.enableAllClasses();
	    result.enable(Capability.MISSING_CLASS_VALUES);
	    result.enable(Capability.NO_CLASS);
		
		return result;
	}

	


	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\t Index of string attribute.\n"
				+ "\t(default: " + this.textIndex + ")", "I", 1, "-I"));		

		result.addElement(new Option("\t Prefix of attributes.\n"
				+ "\t(default: " + this.prefix + ")", "P", 1, "-P"));


		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}


	/**
	 * returns the options of the current setup
	 * 
	 * @return the current options
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-I");
		result.add("" + this.getTextIndex());

		result.add("-P");
		result.add("" + this.getPrefix());


		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}


	/**
	 * Parses the options for this object.
	 * <p/>
	 * 
	 * <!-- options-start --> <!-- options-end -->
	 * 
	 * @param options
	 *            the options to use
	 * @throws Exception
	 *             if setting of options fails
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String textIndexOption = Utils.getOption('I', options);
		if (textIndexOption.length() > 0) {
			String[] textIndexSpec = Utils.splitOptions(textIndexOption);
			if (textIndexSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid index");
			}
			int index = Integer.parseInt(textIndexSpec[0]);
			this.setTextIndex(index);

		}

		String prefixOption = Utils.getOption('P', options);
		if (prefixOption.length() > 0) {
			String[] prefixSpec = Utils.splitOptions(prefixOption);
			if (prefixSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid prefix");
			}
			String pref = prefixSpec[0];
			this.setPrefix(pref);

		}


		super.setOptions(options);

		Utils.checkForRemainingOptions(options);


	}



	/* To allow determineOutputFormat to access to entire dataset
	 * (non-Javadoc)
	 * @see weka.filters.SimpleBatchFilter#allowAccessToFullInputFormat()
	 */
	public boolean allowAccessToFullInputFormat() {
		return true;
	}

	/* Calculates the vocabulary and the word vectors from an Instances object
	 * The vocabulary is only extracted the first time the filter is run.
	 * 
	 */	 
	public void computeWordVecsAndVoc(Instances inputFormat) {

		// The vocabulary is created only in the first execution
		if (!this.isFirstBatchDone())
			this.vocDocFreq = new TreeMap<String, Integer>();

		this.wordVecs = new ArrayList<Map<String, Integer>>();

		// reference to the content of the tweet
		Attribute attrCont = inputFormat.attribute(this.textIndex);

		for (ListIterator<Instance> it = inputFormat.listIterator(); it
				.hasNext();) {
			Instance inst = it.next();
			String content = inst.stringValue(attrCont);


			// tokenises the content
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

			Attribute a = new Attribute(this.prefix + word);

			att.add(a); // adds an attribute for each word using a prefix

			// pw.println("word: " + word + " bytes: "
			// + Arrays.toString(word.getBytes()) + " attribute name: "
			// + a.name() + " HashValue:" + this.vocDocFreq.get(word));

		}

		Instances result = new Instances(inputFormat.relationName(), att, 0);

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
				if (result.attribute(this.prefix + word) != null)
					values[result.attribute(this.prefix + word).index()] = wordVec
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


	public int getTextIndex() {
		return textIndex;
	}


	public void setTextIndex(int textIndex) {
		this.textIndex = textIndex;
	}


	public String getPrefix() {
		return prefix;
	}


	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}

	public static void main(String[] args) {
		runFilter(new TwitterNlpWordToVector(), args);
	}

}
