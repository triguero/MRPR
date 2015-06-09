
/*
	stratPG.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package org.apache.mahout.keel.Algorithms.Instance_Generation.stratPG;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSSMASFLSDE;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Chen.ChenGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ENPC.ENPCGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.HYB.HYBGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ICPL.ICPLGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDE.IPLDEGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS.IPLDECSGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.PSO.PSOGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.RSP.RSPGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.SSMASFLSDE.SSMASFLSDE;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import java.util.*;

import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.*;

import org.core.*;

import org.core.*;

import java.util.StringTokenizer;



/**
 * @param k Number of neighbors
 * @param Population Size.
 * @param ParticleSize.
 * @param Scaling Factor.
 * @param Crossover rate.
 * @param Strategy (1-5).
 * @param MaxIter
 * @author Isaac Triguero
 * @version 1.0
 */
public class stratPGGenerator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int StrataSize = 10000;
  private int stratos = 0;
  private String PGmethod = "IPADE";
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;


  
  private boolean valid;
	protected int asigned [];
	protected int asign [];
  
  /**
   * Build a new stratPGGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public stratPGGenerator(PrototypeSet _trainingDataSet, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="stratPG";
      

  }
  


  /**
   * Build a new stratPGGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public stratPGGenerator(PrototypeSet t, Parameters parameters)
  {
      super(t, parameters);
      algorithmName="stratPG";
      this.k =  parameters.getNextAsInt();
      this.StrataSize =  parameters.getNextAsInt();//*trainingDataSet.get(0).numberOfInputs(); //NC*1000
      this.PGmethod =  parameters.getNextAsString();

      
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      System.out.print("\nIsaac dice:  " + k + "Strata = "+ this.StrataSize + "\n");
      //numberOfPrototypes = getSetSizeFromPercentage(parameters.getNextAsDouble());
  }
  

  
  public void computeStrata(){
		
		valid=true;
		asign = new int [trainingDataSet.size()];
		asigned = new int [this.stratos];
		int counters [] =new int [this.numberOfClass]; // array de contadores, 
		
	    Arrays.fill(asigned, 0);
	    Arrays.fill(counters, 0);
		
		for(int i=0; i<trainingDataSet.size();i++){
			
			asign[i]=counters[(int)trainingDataSet.get(i).getOutput(0)];   // 
			asigned[counters[(int)trainingDataSet.get(i).getOutput(0)]]++;  // el estrato tiene una instancia más..
			
			counters[(int)trainingDataSet.get(i).getOutput(0)]++;
			counters[(int)trainingDataSet.get(i).getOutput(0)]=counters[(int)trainingDataSet.get(i).getOutput(0)]%this.stratos;
		}	
		
		for(int i=0; i<this.stratos;i++){
			if(asigned[i]<2){
				valid=false;
			}
			System.out.println(i+": "+asigned[i]);
		}
	}
	
    
  
	public PrototypeSet[] prepareStratas(){
		
		computeStrata();
		
		PrototypeSet strato[] = new PrototypeSet[this.stratos];
		
		for(int i=0; i<this.stratos; i++){
			strato[i] = new PrototypeSet();
		}
		
	    for(int i=0; i<trainingDataSet.size(); i++){
	    	
	    	strato[asign[i]].add(new Prototype(trainingDataSet.get(i)));
	    	
	    }
	    
	    return strato;
  }
	
	

  /**
   * Generate a reduced prototype set by the stratPGGenerator method.
   * @return Reduced set by stratPGGenerator's method.
 * @throws Exception 
   */
  
  
  public PrototypeSet reduceSet() throws Exception
  {
	  System.out.print("\nThe stratification process is is starting...\n Computing...\n");
	  
	  this.stratos = trainingDataSet.size()/this.StrataSize;
	
	/*  
	  for(int j=0; j<this.numberOfClass; j++){
		  System.out.println("from class: "+j+ ", "+ (trainingDataSet.getFromClass(j)).size());
	  }
	  
	  */
	  
	  //System.out.println("Training data: "+ trainingDataSet.size());
	  System.out.println("Número de estratos: "+ this.stratos);
	  
	  PrototypeSet strata[] = prepareStratas();
	  
	  /*
	  for(int i=0; i<this.stratos; i++){
		  System.out.println("strata size =" + strata[i].size());
		  
		  for(int j=0; j<this.numberOfClass; j++){
			  System.out.println("from class: "+j+ ", "+ (strata[i].getFromClass(j)).size());
		  }

		  System.out.println("***********");
	  }
	  */

	  
	  // A PG method for each strata[i]
	  PrototypeSet reduced[] = new PrototypeSet[this.stratos];
	  
	  for(int i=0; i< this.stratos; i++)
	  {
		  
		  if(this.PGmethod.equalsIgnoreCase("IPADE")){
			  IPLDEGenerator algorithm = new IPLDEGenerator(strata[i], 1, 10000, 8, 20, 0.5, 0.9, 0.03, 0.07);
			  
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
			  
		  }else if (this.PGmethod.equalsIgnoreCase("IPADECS")){
			  
			  IPLDECSGenerator algorithm = new IPLDECSGenerator(strata[i], 1, 10, 500, 8, 20, 0.1, 0.9, 0.1,0.1,  0.03, 0.07, 3);
			  
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
			  
		  }else if (this.PGmethod.equalsIgnoreCase("SSMASFLSDE")){
			  HandlerSSMASFLSDE algorithm = new HandlerSSMASFLSDE();
			  
			//  strata[i].print();
			  
			  reduced[i] = new PrototypeSet(algorithm.ejecutar(strata[i].toInstanceSet(), strata[i]));
			  

		  }else if (this.PGmethod.equalsIgnoreCase("PSO")){
			  PSOGenerator algorithm = new PSOGenerator(strata[i], 1,50, 2, 500, 1, 3, 0.25, 1.5, 0.5);
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
			  
		  }else if (this.PGmethod.equalsIgnoreCase("RSP3")){
			  RSPGenerator algorithm = new  RSPGenerator(strata[i], 0, "diameter");
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
			  
		  }else if (this.PGmethod.equalsIgnoreCase("ICPL2")){
			  ICPLGenerator algorithm = new  ICPLGenerator(strata[i], 2, "RT2", 3, 1);
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
		  
		  }else if (this.PGmethod.equalsIgnoreCase("ENPC")){
			  ENPCGenerator algorithm = new  ENPCGenerator(strata[i], 3, 250);
			  reduced[i] = new PrototypeSet(algorithm.reduceSet().clone());
		  
		  }
		  
		  
	  }	  
	  
	  
	  //Combining the solution: Only joined.
	  
	  PrototypeSet solucion = new PrototypeSet();
	  for(int i=0; i< this.stratos; i++)
	  {
		  solucion.add(reduced[i].clone());
	  }

	  
	  //solucion.print();
	  PrototypeSet nominalPopulation = new PrototypeSet();
      nominalPopulation.formatear(solucion);
	  double trialFitness= accuracy(nominalPopulation,trainingDataSet);
	 
	  System.out.println("Final Fitness = "+ trialFitness);
	  System.out.println("Reduction %, result set = "+((trainingDataSet.size()-solucion.size())*100.)/trainingDataSet.size()+ "\n");
	  

     return solucion;
  }
  
  /**
   * General main for all the prototoype generators
   * @param args Arguments of the main function.
 * @throws Exception 
   */
  public static void main(String[] args) throws Exception
  {
      Parameters.setUse("stratPG", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
      Parameters.assertBasicArgs(args);
      
      PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
      PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
      
      
      long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
      stratPGGenerator.setSeed(seed);
      
      int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
      int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
      int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
      int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);

      
      //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
     //System.out.print(" swarm ="+swarm+"\n");
      
      
      stratPGGenerator generator = new stratPGGenerator(training, k,swarm,particle,iter, 0.5,0.5,1);
      
  	  
      PrototypeSet resultingSet = generator.execute();
      
  	//resultingSet.save(args[1]);
      //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
      int accuracy1NN = KNN.classficationAccuracy(resultingSet, test);
      generator.showResultsOfAccuracy(Parameters.getFileName(), accuracy1NN, test);
  }

}
