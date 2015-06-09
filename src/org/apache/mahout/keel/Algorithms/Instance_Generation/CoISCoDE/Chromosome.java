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

import org.core.Randomize;

public class Chromosome implements Comparable{
	
	private int body[];
	private int size;
	private double fitness;
	
	private static int diff[];
	
	/**
	 * Clone method
	 */
	@Override
	public Chromosome clone(){
		
		Chromosome clon=new Chromosome();
		
		clon.fitness=fitness;
		clon.size=size;
		
		clon.body=new int [body.length];
		
		System.arraycopy(body,0,clon.body, 0, body.length);
		
		return clon;
	}
	
	public Chromosome(){
		
	}
	
	public Chromosome(int size, double init){
	
		body=new int [size];
		
		for(int i=0;i<size;i++){
			if(Randomize.Rand()<init){
				body[i]=1;
			}
			else{
				body[i]=0;
			}
		}
		
		this.size=size;
		fitness=-1.0;
	}

	
	public void setBody(int newBody[]){	
		
		body=new int [newBody.length];
		
		System.arraycopy(newBody,0,body, 0, body.length);
		
	}
	
	public int [] getBody(){
		
		int newBody[];
		
		newBody=new int [body.length];
		
		System.arraycopy(body,0,newBody, 0, body.length);
		
		
		return newBody;
	}
	
	public int get(int pos){
		
		return body[pos];
	
	}

	public void set(int pos,int value){
		
		body[pos]=value;
	
	}
	
	public void setFitness(double value){
		
		fitness= value;
	}
	
	public double getFitness(){
		
		return fitness;
	}
	
	public double computeReduction(){
		
		double ones=0.0;
		
		for(int i=0;i<size;i++){
			if(body[i]==1){
				ones+=1.0;
			}
		}
		
		ones/=size;
		
		return 1.0-ones;
	}
	
	/**
	 * Classic HUX cross operator
	 * 
	 * @param second Second chromosome to cross
	 * 
	 * @return Offspring
	 */
	public Chromosome HUX(Chromosome second){
		
		int index=0;
		int aux;
		diff=new int [size];
		
		for(int i=0;i<size;i++){
			
			if(body[i]!=second.body[i]){
				diff[index]=i;
				index++;
			}
		}
		
		shuffleDiff(index);
		
		index=index/2;
	
		for(int i=0;i<index;i++){
			if(Randomize.Randdouble(0.0, 1.0)<0.2){ //0.25
				aux=body[diff[i]];
				body[diff[i]]=second.body[diff[i]];
				second.body[diff[i]]=aux;
			}
			else{
				body[diff[i]]=0;
				second.body[diff[i]]=0;
			}
		}
		fitness=-1.0;
		second.fitness=-1.0;
		
		return second;
	}
	
	/**
	 * Shuffles the vector of different alleles
	 * 
	 * @param size Size of the vector
	 */
	private void shuffleDiff(int size){
		
		int pos,tmp;
		
	    for (int i=0; i<size; i++) {
	    	
	    	pos = Randomize.Randint (0, size);
	    	tmp = diff[i];
	    	diff[i] = diff[pos];
	    	diff[pos] = tmp;
	    }
	}
	
	/**
	 * Compare to method
	 */
	@Override
	public int compareTo(Object o) {
		
		Chromosome other= (Chromosome)o;
		
		if(this.fitness>other.fitness){
			return -1;
		}
		
		if(this.fitness<other.fitness){
			return 1;
		}
		
		return 0;
	}

}//end-class
