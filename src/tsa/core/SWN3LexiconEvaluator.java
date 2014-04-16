/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tsa.core;

/**
 *
 * 
 * 
 * Based on the following example:
 * https://code.google.com/p/lukejia-svn/source/browse/trunk/sfe-fyp/src/ie/dit/comp/lukejia/fyp/swn/SWN3.java?r=96&spec=svn96
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

public class SWN3LexiconEvaluator extends LexiconEvaluator{

	public SWN3LexiconEvaluator(String file) {
		super(file);
	}


	public void processDict() throws IOException {

		HashMap<String, Double> _temp = new HashMap<String, Double>();
		BufferedReader bf=new BufferedReader(new FileReader(this.path));
		String line = "";

		while ((line = bf.readLine()) != null) {
			if (line.startsWith("#") || line.startsWith("				#")) {
				continue;
			}

			String[] data = line.split("\t");
			Double score = Double.parseDouble(data[2])
					- Double.parseDouble(data[3]);
			String[] words = data[4].split(" ");
			for (String w : words) {
				String[] w_n = w.split("#");
				// w_n[0] += "#" + data[0];
				// int index = Integer.parseInt(w_n[1]) - 1;
				if (_temp.containsKey(w_n[0])) {
					Double v = _temp.get(w_n[0]);
					_temp.put(w_n[0], (v + score) / 2);
				} else {
					_temp.put(w_n[0], score);
				}
			}
		}

		Set<String> temp = _temp.keySet();
		for (Iterator<String> iterator = temp.iterator(); iterator.hasNext();) {
			String word = (String) iterator.next();
			double score = _temp.get(word);

			dict.put(word, String.valueOf(score));
		}

		bf.close();

	}



	static public void main(String args[]) throws IOException {
		LexiconEvaluator s = new SWN3LexiconEvaluator("lexicons/SentiWordNet_3.0.0.txt");
		s.processDict();

		for (String word : s.getDict().keySet()) {
			System.out.println(word + " " + s.retrieveValue(word));
		}

	}
}