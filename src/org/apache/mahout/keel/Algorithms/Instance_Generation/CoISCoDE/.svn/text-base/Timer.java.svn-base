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

public class Timer{
	
	private static long initialTime;
	
	private static double modelTime;
	private static double trainingTime;
	private static double testTime;
	
	/** 
	 * Resets the time counter
	 * 
	 */
	public static void resetTime(){
		
		initialTime = System.currentTimeMillis();
		
	}//end-method
	
	/** 
	 * Set model time
	 * 
	 */
	public static void setModelTime(){
		
		modelTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
		
	}//end-method
	
	/** 
	 * Set training time
	 * 
	 */
	public static void setTrainingTime(){
		
		trainingTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
		
	}//end-method
	
	/** 
	 * Sets training time
	 * 
	 */
	public static void setTestTime(){
		
		testTime=((double)System.currentTimeMillis()-initialTime)/1000.0;
		
	}//end-method
	
	/**
	 * Get model time
	 * 
	 * @return Model time
	 */
	public static double getModelTime(){
		
		return modelTime;
		
	}//end-method
	
	/**
	 * Get training time
	 * 
	 * @return Training time
	 */
	public static double getTrainingTime(){
		
		return trainingTime;
		
	}//end-method
	
	/**
	 * Get test time
	 * 
	 * @return Test time
	 */
	public static double getTestTime(){
		
		return testTime;
		
	}//end-method

}//end-class
