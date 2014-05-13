package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.SingleClassifierEnhancer;

import weka.core.Instances;
import weka.core.Instance;

public class ClassifiersConditionalOnAttribute extends SingleClassifierEnhancer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7607524481740473367L;

	private static int ATTINDEX = 1;

	private Classifier[] m_Classifiers = null;

	public void buildClassifier(Instances trainingData) throws Exception {

		Instances[] datasets = new Instances[trainingData.attribute(ATTINDEX)
				.numValues()];
		for (int i = 0; i < datasets.length; i++) {
			datasets[i] = new Instances(trainingData,
					trainingData.numInstances());
		}
		for (int i = 0; i < trainingData.numInstances(); i++) {
			datasets[(int) trainingData.instance(i).value(ATTINDEX)]
					.add(trainingData.instance(i));
		}
		m_Classifiers = AbstractClassifier.makeCopies(m_Classifier,
				datasets.length);
		for (int i = 0; i < datasets.length; i++) {
			m_Classifiers[i].buildClassifier(datasets[i]);
		}
	}

	public double[] distributionForInstance(Instance testInstance)
			throws Exception {

		return m_Classifiers[(int) testInstance.value(ATTINDEX)]
				.distributionForInstance(testInstance);
	}

	public String toString() {

		if (m_Classifiers == null) {
			return "No model built yet.";
		} else {
			String result = "";
			for (Classifier c : m_Classifiers) {
				result += c + "\n\n";
			}
			return result;
		}
	}

	public static void main(String[] arguments) {

		runClassifier(new ClassifiersConditionalOnAttribute(), arguments);
	}
}