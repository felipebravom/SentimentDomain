package simulator;

import java.util.HashMap;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import tsa.core.LexiconEvaluator;


/**
 * Lexicon Change
 *
 * @author Felipe Bravo-Marquez (fjb11@students.waikato.ac.nz)
 * @version $Revision: 1 $
 */



public class LexiconChange {

	protected LexiconEvaluator lex;

	public LexiconChange(LexiconEvaluator lex) {
		this.lex=lex;
	}

	static public int getRandom(int min, int max){
		Random r = new Random();
		int i1 = r.nextInt(max - min + 1) + min;
		return  i1;
	}

	public void createUniformNoise(double lambda,int min, int max){

		Map<String,String> dict=this.lex.getDict();
		for(String word:dict.keySet()){
			double actScore=Double.parseDouble(dict.get(word));
			double newScore=lambda*getRandom(min,max)+(1-lambda)*actScore;
			dict.put(word, Double.toString(Math.round(newScore)));			
		}

	}


	static public void main(String args[]) throws IOException{
		LexiconEvaluator l2 = new LexiconEvaluator("lexicons/AFINN-111.txt");
		l2.processDict();
			

		System.out.println(l2.retrieveValue("wrong"));
		System.out.println(l2.retrieveValue("happy"));
		System.out.println(l2.retrieveValue("good"));
		
		
		for(int i=0;i<1000;i++){
			
			
			System.out.println("CHANGE\n");
			
			LexiconChange lexChange=new LexiconChange(l2);
			lexChange.createUniformNoise(0.6, -5, 5);
			System.out.println(l2.retrieveValue("wrong"));
			System.out.println(l2.retrieveValue("happy"));
			System.out.println(l2.retrieveValue("good"));
	
		}
				
		
	}



}

