/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ClassifierPerCluster.java
 * Copyright (C) 2014 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.meta;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.SparseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Randomizable;

/**
 * <!-- globalinfo-start --> <!-- globalinfo-end -->
 * 
 * <!-- options-start --> * <!-- options-end -->
 * 
 * @author Eibe Frank
 * @author fracpete
 * @version $Revision: 10331 $
 */
public class ClassifierPerCluster extends RandomizableSingleClassifierEnhancer {

	/** for serialization */
	private static final long serialVersionUID = -8687069451420259138L;

	/** the cluster algorithm used (template) */
	protected Clusterer m_Clusterer = new SimpleKMeans();

	/** the modified training data header */
	protected Instances m_ClusteringHeader;

	/** the classifiers that have been built */
	protected Classifier[] m_Classifiers;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Finds clusters using the given clusterer and builds one classifier"
				+ " per cluster using the given classification algorithm. The seed for the random "
				+ " number generator is ignored if the clusterer is not randomizable.";
	}

	/**
	 * Gets an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tFull name of clusterer.\n"
				+ "\t(default: " + defaultClustererString() + ")", "C", 1, "-C"));

		result.addAll(Collections.list(super.listOptions()));

		result.addElement(new Option("", "", 0,
				"\nOptions specific to clusterer "
						+ m_Clusterer.getClass().getName() + ":"));

		result.addAll(Collections.list(((OptionHandler) m_Clusterer)
				.listOptions()));

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

		result.add("-C");
		result.add("" + getClustererSpec());

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

		String clustererString = Utils.getOption('C', options);
		if (clustererString.length() > 0) {
			String[] clustererSpec = Utils.splitOptions(clustererString);
			if (clustererSpec.length == 0) {
				throw new IllegalArgumentException(
						"Invalid clusterer specification string");
			}
			String clustererName = clustererSpec[0];
			clustererSpec[0] = "";
			setClusterer((Clusterer) Utils.forName(Clusterer.class,
					clustererName, clustererSpec));
		} else {
			setClusterer(AbstractClusterer.forName(defaultClustererString(),
					null));
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * String describing default clusterer.
	 * 
	 * @return the classname
	 */
	protected String defaultClustererString() {
		return SimpleKMeans.class.getName();
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String clustererTipText() {
		return "The clusterer to be used.";
	}

	/**
	 * Set the base clusterer.
	 * 
	 * @param value
	 *            the clusterer to use.
	 */
	public void setClusterer(Clusterer value) {
		m_Clusterer = value;
	}

	/**
	 * Get the clusterer used as the base learner.
	 * 
	 * @return the current clusterer
	 */
	public Clusterer getClusterer() {
		return m_Clusterer;
	}

	/**
	 * Gets the filter specification string, which contains the class name of
	 * the filter and any options to the filter
	 * 
	 * @return the filter string.
	 */
	protected String getClustererSpec() {

		Clusterer c = getClusterer();
		if (c instanceof OptionHandler) {
			return c.getClass().getName() + " "
					+ Utils.joinOptions(((OptionHandler) c).getOptions());
		}
		return c.getClass().getName();
	}

	/**
	 * Returns the class distribution for the given instance.
	 * 
	 * @throws Exception
	 *             if an error occurred during the prediction
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		// Render instance digestable for clusterer
		double[] values = new double[m_ClusteringHeader.numAttributes()];
		int n = 0;
		for (int i = 0; i < instance.numAttributes(); i++) {
			if (i == instance.classIndex()) {
				continue;
			}
			values[n] = instance.value(i);
			n++;
		}

		// We want to create a sparse instance if the input is sparse
		Instance newInst;
		if (instance instanceof DenseInstance) {
			newInst = new DenseInstance(instance.weight(), values);
		} else if (instance instanceof SparseInstance) {
			newInst = new SparseInstance(instance.weight(), values);
		} else {
			throw new Exception("Unknown instance type");
		}

		newInst.setDataset(m_ClusteringHeader);

		// Apply the clusterer and the classifier
		return m_Classifiers[m_Clusterer.clusterInstance(newInst)]
				.distributionForInstance(instance);
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {

		Capabilities result = (Capabilities) m_Clusterer.getCapabilities()
				.clone();
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.and(m_Classifier.getCapabilities());

		return result;
	}

	/**
	 * Builds the classifier
	 * 
	 * @param data
	 *            the training instances
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		// build clusterer
		Instances clusterData = new Instances(data);
		clusterData.setClassIndex(-1);
		clusterData.deleteAttributeAt(data.classIndex());
		if (m_Clusterer instanceof Randomizable) {
			((Randomizable) m_Clusterer).setSeed(getSeed());
		} else {
			if (m_Debug) {
				System.err
						.println("Clusterer does not implement Randomizable.");
			}
		}
		m_Clusterer.buildClusterer(clusterData);
		m_ClusteringHeader = new Instances(clusterData, 0);

		// collect data for each cluster and build classifiers
		Instances[] perClusterData = new Instances[m_Clusterer
				.numberOfClusters()];
		for (int i = 0; i < perClusterData.length; i++) {
			perClusterData[i] = new Instances(data, data.numInstances());
		}
		for (int i = 0; i < clusterData.numInstances(); i++) {
			perClusterData[m_Clusterer.clusterInstance(clusterData.instance(i))]
					.add(data.instance(i));
		}
		for (int i = 0; i < perClusterData.length; i++) {
			perClusterData[i].compactify();
		}
		m_Classifiers = AbstractClassifier.makeCopies(m_Classifier,
				perClusterData.length);
		for (int i = 0; i < perClusterData.length; i++) {
			m_Classifiers[i].buildClassifier(perClusterData[i]);
		}
	}

	/**
	 * Returns a string representation of the classifier.
	 * 
	 * @return a string representation of the classifier.
	 */
	@Override
	public String toString() {

		if (m_Classifiers == null) {
			return "ClassifierPerCluster: no model has been built yet.";
		} else {
			StringBuffer result = new StringBuffer();

			// output clusterer
			result.append("Clustering model:\n\n");
			result.append(m_Clusterer + "\n");

			// output classifiers
			for (int i = 0; i < m_Classifiers.length; i++) {
				result.append("Classifier " + (i + 1) + ":\n\n");
				result.append(m_Classifiers[i] + "\n");
			}
			return result.toString();
		}
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10331 $");
	}

	/**
	 * Runs the classifier with the given options
	 * 
	 * @param args
	 *            the commandline options
	 */
	public static void main(String[] args) {
		runClassifier(new ClassifierPerCluster(), args);
	}
}
