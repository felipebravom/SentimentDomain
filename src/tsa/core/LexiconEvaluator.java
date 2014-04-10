package tsa.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 
 * @author fbravo Evaluates a lexicon from a csv file
 */
public class LexiconEvaluator {

	protected String path;
	protected Map<String, String> dict;

	public LexiconEvaluator(String file) {
		this.dict = new HashMap<String, String>();
		this.path = file;

	}

	public void processDict() throws IOException {
		// first, we open the file
		BufferedReader bf = new BufferedReader(new FileReader(this.path));
		String line;
		while ((line = bf.readLine()) != null) {
			String pair[] = line.split("\t");
			this.dict.put(pair[0], pair[1]);

		}
		bf.close();

	}

	// returns the score associated to a word
	public String retrieveValue(String word) {
		if (!this.dict.containsKey(word)) {
			return "not_found";
		} else {
			return this.dict.get(word);
		}

	}

	public Map<String, String> getDict() {
		return this.dict;
	}

	// counts positive and negative words from a polarity-oriented lexicon
	public Map<String, Integer> evaluatePolarityLexicon(List<String> tokens) {

		Map<String, Integer> sentCount = new HashMap<String, Integer>();

		int negCount = 0;
		int posCount = 0;

		for (String w : tokens) {
			String pol = this.retrieveValue(w);
			if (pol.equals("positive")) {
				posCount++;
			} else if (pol.equals("negative")) {
				negCount++;
			}
		}

		sentCount.put("posCount", posCount);
		sentCount.put("negCount", negCount);

		return sentCount;
	}

	// computes scores from strength-oriented lexicons
	public Map<String, Double> evaluateStrengthLexicon(List<String> tokens) {

		Map<String, Double> strengthScores = new HashMap<String, Double>();
		double posScore = 0;
		double negScore = 0;
		for (String w : tokens) {
			String pol = this.retrieveValue(w);
			if (!pol.equals("not_found")) {
				double value = Double.parseDouble(pol);
				if (value > 0) {
					posScore += value;
				} else {
					negScore += value;
				}
			}
		}
		strengthScores.put("posScore", posScore);
		strengthScores.put("negScore", negScore);

		return strengthScores;
	}

	static public void main(String args[]) throws IOException {

		LexiconEvaluator l = new LexiconEvaluator("lexicons/opinion-finder.txt");
		l.processDict();
		System.out.println(l.retrieveValue("wrong"));
		System.out.println(l.retrieveValue("happy"));
		System.out.println(l.retrieveValue("good"));

		LexiconEvaluator l2 = new LexiconEvaluator("lexicons/AFINN-111.txt");
		l2.processDict();
		System.out.println(l2.retrieveValue("wrong"));
		System.out.println(l2.retrieveValue("happy"));
		System.out.println(l2.retrieveValue("good"));

		LexiconEvaluator l5 = new LexiconEvaluator(
				"lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt");
		l5.processDict();
		System.out.println(l5.retrieveValue("wath"));
		System.out.println(l5.retrieveValue("hate"));
		System.out.println(l5.retrieveValue("good"));

		LexiconEvaluator l3 = new LexiconEvaluator(
				"lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt");
		l3.processDict();
		System.out.println(l3.retrieveValue("love"));
		System.out.println(l3.retrieveValue("sad"));
		System.out.println(l3.retrieveValue("sick"));

		LexiconEvaluator l4 = new LexiconEvaluator("lexicons/BingLiu.csv");
		l4.processDict();
		System.out.println(l4.retrieveValue("love"));
		System.out.println(l4.retrieveValue("hate"));
		System.out.println(l4.retrieveValue("sick"));

	}
}
