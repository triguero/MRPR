/**
 * 
 * File: Timer.java
 * 
 * Auxiliar class to support timing reports in model+training+test algorithms
 * 
 * @author Written by Joaquín Derrac (University of Granada) 20/04/2010 
 * @version 1.0 
 * @since JDK1.5
 * 
 */
package org.apache.mahout.keel.Algorithms.Instance_Generation.CoISCoDE;

import java.util.Arrays;

public class Translator{
	
	private static int translations[];
	private static int reverse[];
	private static int nextFree;
	
	public static void initialize(){
	
		translations=new int [100];
		
		Arrays.fill(translations, -1);
		
		reverse=new int [100];
		
		Arrays.fill(translations, -1);
		
		nextFree=0;
		
	}
	
	public static void learnValue(int val){
		
		if(translations[val]==-1){
			translations[val]=nextFree;
			reverse[nextFree]=val;
			nextFree++;
		}
		
	}
	
	public static int real2safe(int val){
		
		return translations[val];
		
		
	}
	
	public static int safe2real(int val){
		
		return reverse[val];
	}
	
	public static int getSafeClasses(){
		return nextFree;
	}
	
	

}//end-class
