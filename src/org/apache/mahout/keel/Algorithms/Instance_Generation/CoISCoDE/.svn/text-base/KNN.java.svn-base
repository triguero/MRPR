/**
 * 
 * File: IFWS.java
 * 
 * The Cam NN Algorithm.
 * It makes use of Cam distance to improve the KNN classification. 
 * 
 * @author Written by Joaqu�n Derrac (University of Granada) 13/11/2008 
 * @version 1.0 
 * @since JDK1.5
 * 
 */

package org.apache.mahout.keel.Algorithms.Instance_Generation.CoISCoDE;

import java.util.Arrays;

class KNN{

	private static double data[][];
	private static int output[];
	
	private static int nearestN[];
	private static double minDist[];
	
	private static int K;
	private static int nClasses;
	private static int selectedClasses[];
	private static double dist;
	
	private static int prediction;
	private static int predictionValue;
	
	private static int IS[]; 
	
	public static void setData(double newData[][],int newOutput[]){	
		
		data=new double [newData.length][newData[0].length];
		output=new int [data.length];
		
		int pointer=0;
		
		for(int c=0;c<nClasses;c++){		
			for(int i=0;i<newData.length;i++){
				if(newOutput[i]==c){
					System.arraycopy(newData[i],0,data[pointer], 0, newData[i].length);
					output[pointer]=c;
					pointer++;
				}
			}
		}
		
		IS=new int [data.length];
		
		Arrays.fill(IS, 1);

	}
	
	public static void setIS(int [] values){
		
		for(int i=0;i<IS.length;i++){
			IS[i]=values[i];
		}
		
	}
	
	
	public static void setK(int value){
		
		K=value;
		
		nearestN = new int[K];
		minDist = new double[K];
	}
	
	public static void setClasses(int value){
		
		nClasses=value;
		
		selectedClasses= new int[nClasses];
	}
    
	public static int classifyTestInstance(double example[]){
	
		int output;
		
		if(K==1){
			output=test1NN(example);
		}
		else{
			output=testKNN(example);
		}
		
		return output;
	}
	
	public static int classifyTrainInstance(int index){
		
		int output;
		
		if(K==1){
			output=train1NN(index);
		}
		else{
			output=trainKNN(index);
		}
		
		return output;
	}
	
	
	private static int test1NN(double example[]){
	
	    nearestN[0]=0;
	    minDist[0]=Double.MAX_VALUE;
	    
	    //1NN Method starts here
	    
		for (int i=0; i<data.length; i++) {
		
			if(IS[i]==1){
				dist = squaredEuclideanDistance(example,i);
	
				//see if it's nearer than our previous selected neigbours
				if (dist < minDist[0]) {
				
					minDist[0] = dist;
					nearestN[0] = i;
				
					//System.out.println(output[nearestN[0]]);
				}
				
				
				
			}
		}
		
		return output[nearestN[0]];
	}
	
	private static int train1NN(int index){

		int old;
		
		nearestN[0]=0;
	    minDist[0]=Double.MAX_VALUE;
	    
	    //leave one out
	    old=IS[index];
	    IS[index]=0;
	    
		for (int i=0; i<data.length; i++) {
		
			dist = squaredEuclideanDistance(index,i);
			
			if(IS[i]==1){
				
				//see if it's nearer than our previous selected neigbours
	
				if (dist < minDist[0]) {
				
					minDist[0] = dist;
					nearestN[0] = i;
						
				}
			}
		}
		
		//leave one out
	    IS[index]=old;
		 
		return output[nearestN[0]];

	}
	
	private static int testKNN (double example[]) {

		boolean stop;

		Arrays.fill(nearestN, -1);
		Arrays.fill(minDist, Double.MAX_VALUE);
		
	    //KNN Method starts here
	    
		for (int i=0; i<data.length; i++) {
		
			if(IS[i]==1){
			    dist = squaredEuclideanDistance(example,i);
	
			    //see if it's nearer than our previous selected neighbors
				stop=false;
					
				for(int j=0;j<K && !stop;j++){
					
					if (dist < minDist[j]) {
						    
						for (int l = K - 1; l >= j+1; l--) {
							minDist[l] = minDist[l - 1];
							nearestN[l] = nearestN[l - 1];
						}	
							
						minDist[j] = dist;
						nearestN[j] = i;
						stop=true;
					}
				}
			}
			
		}
		
		//we have check all the instances... see what is the most selected class
		Arrays.fill(selectedClasses, 0);
		
		for (int i=0; i<K; i++) {
			if(nearestN[i]!=-1){
				selectedClasses[output[nearestN[i]]]+=1;
			}
		}
		
		prediction=-1;
		predictionValue=0;
		
		for (int i=0; i<nClasses; i++) {
		    if (predictionValue < selectedClasses[i]) {
		        predictionValue = selectedClasses[i];
		        prediction = i;
		    }
		}
		
		return prediction;
	
	} //end-method	
	
	private static int trainKNN (int index) {

		boolean stop;

		Arrays.fill(nearestN, -1);
		Arrays.fill(minDist, Double.MAX_VALUE);
		
		//leave one out
	    int old=IS[index];
	    IS[index]=0;
	    
	    //KNN Method starts here
	    
		for (int i=0; i<data.length; i++) {
		
			dist = squaredEuclideanDistance(index,i);
			
			if(IS[i]==1){
	
			    //see if it's nearer than our previous selected neighbors
				stop=false;
					
				for(int j=0;j<K && !stop;j++){
					
					if (dist < minDist[j]) {
						    
						for (int l = K - 1; l >= j+1; l--) {
							minDist[l] = minDist[l - 1];
							nearestN[l] = nearestN[l - 1];
						}	
							
						minDist[j] = dist;
						nearestN[j] = i;
						stop=true;
					}
				}
			}
			
		}
		
		//we have check all the instances... see what is the most selected class
		Arrays.fill(selectedClasses, 0);
		
		for (int i=0; i<K; i++) {
			if(nearestN[i]!=-1){
				selectedClasses[output[nearestN[i]]]+=1;
			}
		}
		
		prediction=-1;
		predictionValue=0;
		
		for (int i=0; i<nClasses; i++) {
		    if (predictionValue < selectedClasses[i]) {
		        predictionValue = selectedClasses[i];
		        prediction = i;
		    }
		}
		
		IS[index]=old;
		
		return prediction;
	
	} //end-method	
	
	private static double squaredEuclideanDistance(int a,int b){
		
		double length=0.0;
		double value;

		for (int i=0; i<data[a].length; i++) {
			
			value = data[a][i]-data[b][i];
			length += value*value;
		}
				
		return length;
	}
	
	private static double squaredEuclideanDistance(double example[],int b){
		
		double length=0.0;
		double value;

		for (int i=0; i<data[b].length; i++) {
			
			value = example[i]-data[b][i];
			length += value*value;
		}
				
		return length;
	}

	public static int getTrueOutput(int index){
		
		return output[index];
	}
    
} //end-class 
