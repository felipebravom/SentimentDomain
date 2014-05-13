package weka.clusterers;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.regex.Pattern;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.CharSequenceLowercase;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.StringArrayIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.InstanceList;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

public class LdaCluster extends AbstractClusterer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 864281599421580096L;

	private ParallelTopicModel model; // the model
	private InstanceList instances; // the training data and the pipes
	protected int numberOfTopics=10; // Number of topics
	protected double alphaSum=1.0; 
	protected double beta=0.01;




	/**
	 * Returns a string describing this clusterer
	 * 
	 * @return a description of the evaluator suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "LDA MALLET.";
	}


	@Override
	public Capabilities getCapabilities() {
		Capabilities result;

		result = new Capabilities(this);
		result.enableAll();
		return result;
	}



	/**
	 * Gets an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\t Number of topics.\n"
				+ "\t(default: " + this.numberOfTopics + ")", "N", 1, "-N"));		

		result.addElement(new Option("\t AlphaSum.\n"
				+ "\t(default: " + this.alphaSum + ")", "A", 1, "-A"));

		result.addElement(new Option("\t Beta.\n"
				+ "\t(default: " + this.beta + ")", "B", 1, "-B"));



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

		result.add("-N");
		result.add("" + this.numberOfClusters());

		result.add("-A");
		result.add("" + this.alphaSum);

		result.add("-B");
		result.add("" + this.beta);

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

		String numTopicsOption = Utils.getOption('N', options);
		if (numTopicsOption.length() > 0) {
			String[] numTopicsSpec = Utils.splitOptions(numTopicsOption);
			if (numTopicsSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid number of topics specification");
			}
			int numOfTopics = Integer.parseInt(numTopicsSpec[0]);
			this.setNumberOfTopics(numOfTopics);

		} 


		String alphaSumOption = Utils.getOption('A', options);
		if (alphaSumOption.length() > 0) {
			String[] alphaSumSpec = Utils.splitOptions(alphaSumOption);
			if (alphaSumSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid number of AlphaSum");
			}
			double alphaSumValue = Double.parseDouble(alphaSumSpec[0]);
			this.setAlphaSum(alphaSumValue);

		}

		String betaOption = Utils.getOption('B', options);
		if (betaOption.length() > 0) {
			String[] betaSpec = Utils.splitOptions(betaOption);
			if (betaSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid number of beta");
			}
			double betaValue = Double.parseDouble(betaSpec[0]);
			this.setBeta(betaValue);		


		} 




		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}





	@Override
	public void buildClusterer(Instances data) throws Exception {

		Attribute attrCont = data.attribute("content");

		String[] documents=new String[data.numInstances()];

		for (int i = 0; i < data.numInstances(); i++) {
			documents[i] = data.instance(i).stringValue(attrCont);
		}

		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

		// Pipes: lowercase, tokenize, remove stopwords, map to features
		pipeList.add(new CharSequenceLowercase());
		pipeList.add(new CharSequence2TokenSequence(Pattern
				.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
		pipeList.add(new TokenSequenceRemoveStopwords(new File(
				"stoplists/en.txt"), "UTF-8", false, false, false));
		pipeList.add(new TokenSequence2FeatureSequence());

		this.instances = new InstanceList(new SerialPipes(pipeList));


		this.instances.addThruPipe(new StringArrayIterator(documents)); // data,
		// label,
		// name
		// fields


		// Create a model with 100 topics, alpha_t = 0.01, beta_w = 0.01
		//  Note that the first parameter is passed as the sum over topics, while
		//  the second is the parameter for a single dimension of the Dirichlet prior.
		int numTopics = this.numberOfClusters();
		this.model = new ParallelTopicModel(numTopics, this.alphaSum, this.beta);

		this.model.addInstances(instances);

		// Use two parallel samplers, which each look at one half the corpus and combine
		//  statistics after every iteration.
		this.model.setNumThreads(2);

		// Run the model for 50 iterations and stop (this is for testing only, 
		//  for real applications, use 1000 to 2000 iterations)
		this.model.setNumIterations(150);
		this.model.estimate();

		// TODO Auto-generated method stub

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		Attribute attrCont = instance.dataset().attribute("content");

		String content=instance.stringValue(attrCont);

		InstanceList testing = new InstanceList(this.instances.getPipe());
		testing.addThruPipe(new cc.mallet.types.Instance(content, null, "test instance", null));

		TopicInferencer inferencer = this.model.getInferencer();
		double[] testProbabilities = inferencer.getSampledDistribution(testing.get(0), 10, 1, 5);

		return testProbabilities;

	}

	/**
	 * Returns a string representation of the clusterer.
	 * 
	 * @return a string representation of the clusterer.
	 */
	@Override
	public String toString() {
		return "TODO";		
	}


	public void setNumberOfTopics(int t){
		this.numberOfTopics=t;
	}

	public int getNumberOfTopics(){
		return this.numberOfTopics;
	}

	@Override
	public int numberOfClusters() {
		// TODO Auto-generated method stub
		return this.numberOfTopics;
	}

	public double getAlphaSum(){
		return this.alphaSum;
	}

	public void setAlphaSum(double alphaSum){
		this.alphaSum=alphaSum;
	}

	public double getBeta(){
		return this.beta;
	}

	public void setBeta(double beta){
		this.beta=beta;
	}

}
