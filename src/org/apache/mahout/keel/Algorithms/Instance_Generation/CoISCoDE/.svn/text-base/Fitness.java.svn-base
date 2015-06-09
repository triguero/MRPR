/**
 * 
 * File: Timer.java
 * 
 * Auxiliar class to support timing reports in model+training+test algorithms
 * 
 * @author Written by Joaqu�n Derrac (University of Granada) 20/04/2010 
 * @version 1.0 
 * @since JDK1.5
 * 
 */
package org.apache.mahout.keel.Algorithms.Instance_Generation.CoISCoDE;

class Fitness{
	
	public static double computeFitness(Chromosome individual, int population){
		
		double fitness=0.0;
		double acc,globalAcc,red;
		
		int arrayIS []= new int [CoIS.getTrainSize()];
		int pointer=0;
		
		//generate IS array
		for(int i=0;i<CoIS.getNPops();i++){
			if(i==population){
				for(int j=0;j<individual.getBody().length;j++){
					arrayIS[pointer]=individual.get(j);
					pointer++;
				}
			}else{
				for(int j=0;j<CoIS.getCollaborator(i).getBody().length;j++){
					arrayIS[pointer]=CoIS.getCollaborator(i).get(j);
					pointer++;
				}
			}
		}
		
		KNN.setIS(arrayIS);
		
		int prediction;
		acc=0.0;
		globalAcc=0.0;
		
		for(int i=0;i<CoIS.getTrainSize();i++){
			prediction=KNN.classifyTrainInstance(i);
			if(prediction==KNN.getTrueOutput(i)){
				globalAcc+=1.0;
				if(prediction==population){
					acc+=1.0;
				}
			}
		}
		
		globalAcc/=CoIS.getTrainSize();
		acc/=individual.getBody().length;
		
		red=individual.computeReduction();
		
		//fitness=(0.5*((0.5*acc)+(0.5*globalAcc)))+(0.5*red);
		if(red==1){
			fitness=0;
		}else{
			fitness=(0.4*((0.5*acc)+(0.5*globalAcc)))+(0.6*red);
		}
		
		CoIS.evaluationSpent();
		
		return fitness;
	}
	
}//end-class
