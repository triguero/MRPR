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

import org.core.Randomize;

public class Population{
	
	private Chromosome individuals[];
	private static int size;
	private static double initProb;
	
	private int threshold;
	private int initialThreshold;
	private int tag;
	
	private boolean trueClass;
	
	public Population(int outputVal,int nElements){	
		
		if(nElements>0){
			trueClass=true;
			
			individuals=new Chromosome[size];
			
			for(int i=0;i<size;i++){
				individuals[i]=new Chromosome(nElements,initProb);
			}
			
		}
		else{
			trueClass=false;
		}

		tag=outputVal;
		initialThreshold=nElements/4;
		threshold=initialThreshold;
		
	}

	public static void setSize(int value){
		size=value;
	}
	
	public static void setInitProb(double value){
		initProb=value;
	}
	
	public Chromosome getBest(){
		
		return individuals[0].clone();
		
	}
	
	public Chromosome getRandom(){
		
		int rand= Randomize.Randint(0, size);
		
		return individuals[rand].clone();
		
	}

	
	private void sort(){
		
		Arrays.sort(individuals);
	}
	
	public void doGeneration(){
		
		Chromosome newPopulation [];
		boolean nuevos=false;
		double value;
		

		//baraje de la poblacion
		newPopulation=shuffle(individuals);
		
		//Cruce 
		
		for(int i=0;i<size;i+=2){
		
			if(hamming(newPopulation[i].getBody(),newPopulation[i+1].getBody())/2>threshold){

				//cruce
				HUXCross(newPopulation,i,i+1);
				
				//evaluacion
				value=Fitness.computeFitness(newPopulation[i], tag);
				newPopulation[i].setFitness(value);
				
				value=Fitness.computeFitness(newPopulation[i+1], tag);
				newPopulation[i+1].setFitness(value);
				
				nuevos=true;
			}
		}
		
		//Seleccion selectiva
		if(nuevos){
			nuevos=mergePop(individuals,newPopulation);
		}
		
		//reinicicializacion
		if(nuevos==false){
			
			threshold--;
			
			if(threshold==0){

				//generar
				for(int i=1; i<size;i++){
					individuals[i]=individuals[0].clone();
				}
				
				//mutar

				for(int i=1; i<size;i++){
					for(int j=0;j<individuals[i].getBody().length;j++){
						if(Randomize.Rand()<0.35){
							if(Randomize.Rand()<0.5){
								individuals[i].set(j,1);
							}
							else{
								individuals[i].set(j,0);
							}
						}
					}
					individuals[i].setFitness(-1.0);
				}
				
				//evaluar
				for(int i=1; i<size;i++){
					value=Fitness.computeFitness(individuals[i], tag);
					individuals[i].setFitness(value);
				}
				
				//ordenar
				sort();
				        
				threshold=initialThreshold;
			}
		}

	}
	
	/**
	 * Shuffles the population
	 * 
	 * @param population IS population
	 * 
	 * @return Shuffled population
	 */
	private static Chromosome [] shuffle(Chromosome[] population){
		
		int index[]=new int [size];
		int pos,tmp;
		Chromosome clon []= new Chromosome [size];
		
	    for (int i=0; i<size; i++){
	    	index[i] = i;
	    }

	    for (int i=0; i<size; i++) {    	
	    	pos = Randomize.Randint (0, size);
	    	tmp = index[i];
		    index[i] = index[pos];
		    index[pos] = tmp;
	    }
	 
	    for (int i=0; i<size; i++) {
	    	clon[i]=population[index[i]].clone();
	    }
		
	    return clon;
	}
	
	/**
	 * Merges two population
	 * 
	 * @param population Old population
	 * @param newPopulation New population
	 * 
	 * @return True if any new chromosome has been acepted. False, if not
	 */
	private static boolean mergePop(Chromosome [] population,Chromosome [] newPopulation){
		
		int index=0;
		int taken=0;
		boolean used=false;
		double bestFitness;
		int bestPosition;
		
		Chromosome [] finalPop=new Chromosome [size]; 
		
		bestFitness=-1;
		bestPosition=-1;
	
		for(int i=0;i<newPopulation.length;i++){
			
			if(newPopulation[i].getFitness()>bestFitness){
				bestFitness=newPopulation[i].getFitness();
				bestPosition=i;
			}
		}
		
		while(taken<size){
			
			if(population[index].getFitness()>bestFitness){
				finalPop[taken]=population[index].clone();
				index++;
			}
			else{
				finalPop[taken]=newPopulation[bestPosition].clone();			
				newPopulation[bestPosition].setFitness(-1.0);
				
				bestFitness=-1;
				bestPosition=-1;
				
				for(int i=0;i<newPopulation.length;i++){
					if(newPopulation[i].getFitness()>bestFitness){
						bestFitness=newPopulation[i].getFitness();
						bestPosition=i;
					}
				}
				used=true;
			}
			taken++;
		}
		
		System.arraycopy(finalPop, 0, population, 0, size);
		
		return used;
	}
	
	/**
	 * Computes hamming distance
	 * 
	 * @param a First array
	 * @param b Second array
	 * 
	 * @return Hamming distance
	 */
	private static int hamming (int a[],int b[]){
		
		int dist=0;
		
		for(int i=0;i<a.length;i++){

			if(a[i]!=b[i]){
				dist++;
			}
		}
		
		return dist;		
	}
	
	/**
	 * HUX crossing operator
	 * 
	 * @param population Is population
	 * @param a First chromosome
	 * @param b Second chromosome
	 */
	private static void HUXCross(Chromosome population [],int a,int b){

		population[b]=population[a].HUX(population[b]);
	}
	
	public void evaluatePopulation(){
		
		double value;
		
		for(int i=0;i<size;i++){
			value=Fitness.computeFitness(individuals[i], tag);
			individuals[i].setFitness(value);
		}
	}

}//end-class
