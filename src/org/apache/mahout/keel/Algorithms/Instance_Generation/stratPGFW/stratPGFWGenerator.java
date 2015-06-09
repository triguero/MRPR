
/*
	stratPGFW.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package org.apache.mahout.keel.Algorithms.Instance_Generation.stratPGFW;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSSMAPGFW;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSSMASFLSDE;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Chen.ChenGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ENPC.ENPCGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.HYB.HYBGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ICPL.ICPLGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPADECSFW.IPADECSFWGenerator;
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
public class stratPGFWGenerator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int StrataSize = 10000;
  private int stratos = 0;
  private String PGmethod = "IPADE";
  private String JoinProcedure = "VotingRule";
  private String Voting = "Kernel";
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;


  private int iterFWinitial;
  private int iterFW;
  private int PopulationFW;
  private int MAX_ITER;
  private int Strategy;
  private int iterSFGSS;
  private int iterSFHC;
  private double Fl, Fu;
  private double tau[] = new double[4];
  
  private boolean valid;
	protected int asigned [];
	protected int asign [];
  
  /**
   * Build a new stratPGFWGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public stratPGFWGenerator(PrototypeSet _trainingDataSet, PrototypeSet _test, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="stratPGFW";
      

  }
  


  /**
   * Build a new stratPGFWGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public stratPGFWGenerator(PrototypeSet t, PrototypeSet _test, Parameters parameters)
  {
      super(t, _test, parameters);
      algorithmName="stratPGFW";
      this.k =  parameters.getNextAsInt();
      this.StrataSize =  parameters.getNextAsInt();//*trainingDataSet.get(0).numberOfInputs(); //NC*1000
      this.PGmethod =  parameters.getNextAsString();
      this.JoinProcedure = parameters.getNextAsString();
      this.Voting = parameters.getNextAsString();
      this.MAX_ITER = parameters.getNextAsInt(); // MAX EPOCHs
      this.iterSFGSS =  parameters.getNextAsInt();
      this.iterSFHC =  parameters.getNextAsInt();
      this.Fl = parameters.getNextAsDouble();
      this.Fu = parameters.getNextAsDouble();
      this.tau[0] =  parameters.getNextAsDouble();
      this.tau[1] =  parameters.getNextAsDouble();
      this.tau[2] =  parameters.getNextAsDouble();
      this.tau[3] =  parameters.getNextAsDouble();
      this.Strategy =  parameters.getNextAsInt();
      this.iterFWinitial =  parameters.getNextAsInt();
      this.iterFW =  parameters.getNextAsInt();
      this.PopulationFW =  parameters.getNextAsInt();
      
      
      
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      System.out.print("\nIsaac dice:  " + k + "Strata = "+ this.StrataSize + "\n");
      //numberOfPrototypes = getSetSizeFromPercentage(parameters.getNextAsDouble());
  }
  

  
  
  
  public static Prototype _1nnWeighted(Prototype current, Prototype Weights, PrototypeSet dataSet)
  {
      Prototype nearestNeighbor = dataSet.get(0);
      int indexNN = 0;
      //double minDist = Distance.dSquared(current, nearestNeighbor);
      //double minDist = Distance.euclideanDistance(current, nearestNeighbor);
      double minDist =Double.POSITIVE_INFINITY;
      double currDist;
      int _size = dataSet.size();
    //  System.out.println("****************");
     // current.print();
      for (int i=0; i<_size; i++)
      {
          Prototype pi = dataSet.get(i);
          //if(!current.equals(pi))
          //{
             // double currDist = Distance.dSquared(current, pi);
          
          // Calculating weighted distance:
          
	          double acc = 0.0;
	          for (int j = 0; j < current.numberOfInputs(); j++)
	          {
	              acc += ((current.getInput(j) - pi.getInput(j)) * (current.getInput(j)  - pi.getInput(j)))*Weights.getInput(j);
	          }
	          
           currDist =  acc;
           // Distance.euclideanDistance(pi,current);

          
           if(currDist >0){
              if (currDist < minDist)
              {
                  minDist = currDist;
                 // nearestNeighbor = pi;
                  indexNN =i;
              }
          }
          //}
      }
      
     // System.out.println("Min dist =" + minDist + " Vecino Cercano = "+ indexNN);
      
      return dataSet.get(indexNN);
  }
  
  
  public double classficationAccuracy1NN_weights(PrototypeSet training, Prototype Weights, PrototypeSet test)
  {
	int wellClassificated = 0;
      for(Prototype p : test)
      {
          Prototype nearestNeighbor = _1nnWeighted(p, Weights, training);          
          
          if(p.getOutput(0) == nearestNeighbor.getOutput(0))
              ++wellClassificated;
      }
  
      
      return 100.0* (wellClassificated / (double)test.size());
  }
  
  
  public PrototypeSet NN_weights(PrototypeSet training, Prototype Weights, PrototypeSet test)
  {
	 PrototypeSet clasificado = new PrototypeSet(test.clone());
	
      for(int i=0; i< test.size(); i++)
      {
    	  
          Prototype nearestNeighbor = _1nnWeighted(test.get(i), Weights, training);          
          
          clasificado.get(i).setFirstOutput(nearestNeighbor.getOutput(0));
          

      }
  
      
      return clasificado;
  }
  
  

  /**
   * MUTANT FOR FEATURE WEIGHTIN
   * @param population
   * @param actual
   * @param mejor
   * @param SFi
   * @return
   */
  
public Prototype mutant(Prototype population[], int actual, int mejor, double SFi){
  	  
  	  
  	  Prototype mutant = new Prototype();
  	  Prototype r1,r2,r3,r4,r5, resta, producto, resta2, producto2, result, producto3, resta3;
  	  

  		 
  		mutant = new Prototype();
  	  
  		//We need three differents solutions of actual
  		   
  	   int lista[] = new int[population.length];
       inic_vector_sin(lista,actual);
       desordenar_vector_sin(lista);
  		      
  	  // System.out.println("Lista = "+lista[0]+","+ lista[1]+","+lista[2]);
  	  
       
  	   r1 = population[lista[0]];
  	   r2 = population[lista[1]];
  	   r3 = population[lista[2]];
  	   r4 = population[lista[3]];
  	   r5 = population[lista[4]];
  		   
  			switch(this.Strategy){
  		   	   case 1: // ViG = Xr1,G + F(Xr2,G - Xr3,G) De rand 1
  		   		 resta = r2.sub(r3);
  		   		 producto = resta.mul(SFi);
  		   		 mutant = producto.add(r1);
  		   	    break;
  			   
  		   	   case 2: // Vig = Xbest,G + F(Xr2,G - Xr3,G)  De best 1
  			   		 resta = r2.sub(r3);
  			   		 producto = resta.mul(SFi);
  			   		 mutant = population[mejor].add(producto);
  			   break;
  			   
  		   	   case 3: // Vig = ... De rand to best 1
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = population[mejor].sub(population[actual]);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = population[actual].add(producto);
  			   	   mutant  = result.add(producto2);
  			   		 			   		 
  			   break;
  			   
  		   	   case 4: // DE best 2
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = r3.sub(r4);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = population[mejor].add(producto);
  			   	   mutant  = result.add(producto2);
  			   break;
  			  
  		   	   case 5: //DE rand 2
  		   		   resta = r2.sub(r3); 
  		   		   resta2 = r4.sub(r5);
  		   		 			   		 
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   		
  			   	   result = r1.add(producto);
  			   	   mutant  = result.add(producto2);
  			   	   
    		       break;
    		       
  		   	   case 6: //DE rand to best 2
  		   		   resta = r1.sub(r2); 
  		   		   resta2 = r3.sub(r4);
  		   		   resta3 = population[mejor].sub(population[actual]);
  		   		   
  			   	   producto = resta.mul(SFi);
  			   	   producto2 = resta2.mul(SFi);
  			   	   producto3 = resta3.mul(SFi);
  			   	   
  			   	   result = population[actual].add(producto);
  			   	   result = result.add(producto2);
  			   	   mutant  = result.add(producto3);
    		       break;
    		       
  		   }   
  	   

  	  // System.out.println("********Mutante**********");
  	 // mutant.print();
  	   
  		for(int j=0; j<mutant.numberOfInputs(); j++){
  			if(mutant.getInput(j)<=0.2){                 //Suggested by Salva!
  				mutant.setInput(j, 0);
  			}else if(mutant.getInput(j)>1){
  				mutant.setInput(j, 1);
  			}
  			
  		}
  	
       
 
  	  return mutant;
 }



  
  /**
   * SFGSS local Search.  FOR  Feature Weighting
   * @param population
   * @return
   */
  public Prototype SFGSS(Prototype population[], int actual, int mejor, double CRi, PrototypeSet reduced){
	  double a=0.1, b=1;

	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
	  double phi = (1+ Math.sqrt(5))/5;
	  double scaling;
	  Prototype crossover, resta, producto, mutant;
	  
	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
	  
		  fi1 = b - (b-a)/phi;
		  fi2 = a + (b-a)/phi;
		  
		  fitnessFi1 = lsff(fi1, CRi, population,actual,mejor, reduced);
		  fitnessFi2 = lsff(fi2, CRi,population,actual,mejor, reduced);
		  
		  if(fitnessFi1> fitnessFi2){
			  b = fi2;
		  }else{
			  a = fi1;  
		  }
	  
	  } // End While
	  
	  
	  if(fitnessFi1> fitnessFi2){
		  scaling = fi1;
	  }else{
		  scaling = fi2;
	  }
	  
	  
	  //Mutation:
	  mutant = new Prototype();
	  mutant = mutant(population, actual, mejor, scaling);
   	  
   	  //Crossover
   	  crossover =new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  }
  
  
  
  /**
   * SFHC local search  for FEATURE WEITHING
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   * @param SFi
   * @return
   */
  
  public  Prototype SFHC(Prototype population[], int actual, int mejor, double SFi, double CRi, PrototypeSet reduced){
	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
	  Prototype crossover, resta, producto, mutant;
	  double h= 0.5;
	  
	  
	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
		  		  
		  fitnessFi1 = lsff(SFi-h, CRi, population,actual,mejor, reduced);
		  fitnessFi2 = lsff(SFi, CRi,  population,actual,mejor, reduced);
		  fitnessFi3 = lsff(SFi+h, CRi,  population,actual,mejor, reduced);
		  
		  if(fitnessFi1 >= fitnessFi2 && fitnessFi1 >= fitnessFi3){
			  bestFi = SFi-h;
		  }else if(fitnessFi2 >= fitnessFi1 && fitnessFi2 >= fitnessFi3){
			  bestFi = SFi;
			  h = h/2; // H is halved.
		  }else{
			  bestFi = SFi;
		  }
		  
		  SFi = bestFi;
	  }
	  
	  
	  //Mutation:
	  mutant = new Prototype();
	  mutant = mutant(population, actual, mejor, SFi);
	 
   	  //Crossover
   	  crossover = new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  
  }
  
  
  /**
   * Local Search Fitness Function for feature weighting
   * @param Fi
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   */
  public double lsff(double Fi, double CRi, Prototype population[], int actual, int mejor, PrototypeSet reduced){
	  Prototype resta, producto, mutant;
	  Prototype crossover;
	  double FitnessFi = 0;
	  
	  
	  //Mutation:
	  mutant = new Prototype();
   	  mutant = mutant(population, actual, mejor, Fi);
   	
   	  
   	  //Crossover
   	  crossover =new Prototype(population[actual]);
   	  
	   for(int j=0; j< population[actual].numberOfInputs(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.setInput(j, mutant.getInput(j)); // Overwrite.
		   }
	   }
	   
	   
	   // Compute fitness

       FitnessFi = classficationAccuracy1NN_weights(reduced, crossover ,trainingDataSet);
	   
   	   return FitnessFi;
  }
  
  
  
  
  /**
   * Feature weighting optimization by SFLSDE algorithm
   * @param actual
   * @return
   */
  public double[] FeatureWeighting(PrototypeSet actual, boolean initial, double[] original){
	  
	  	  int iterations =0;
		  int numberOfInputs = actual.get(0).numberOfInputs();
		  
	  	  if(initial){
	  		  iterations = this.iterFWinitial*numberOfInputs;
	  	  }else{
	  		iterations = this.iterFW*numberOfInputs;
	  	  }
	  
		   PrototypeSet nominalPopulation = new PrototypeSet();
	       nominalPopulation.formatear(actual);
	       

		  Prototype [] Population = new Prototype[PopulationFW]; // I use prototype as double[]
		  Prototype mutation[] = new Prototype[PopulationFW];
		  Prototype crossover[] = new Prototype[PopulationFW];
		  
		  // Initialize population: Randomly.
		  
		  Population[0] = new Prototype(numberOfInputs,0);
		  for(int j=0; j<numberOfInputs; j++){
				Population[0].setInput(j, original[j]);    // The first individual is 1.0 (the original)
		  }
		  
		  
		  for(int i=1; i< this.PopulationFW; i++){
			  Population[i]= new Prototype(numberOfInputs,0);
			  
			  for(int j=0; j<numberOfInputs; j++){
				Population[i].setInput(j, RandomGenerator.Rand());
			  }
			  
			 // Population[i].print();
		  }
		  
		  double ScalingFactor[] = new double[this.PopulationFW];
		  double CrossOverRate[] = new double[this.PopulationFW]; // Inside of the Optimization process.
		  double fitness[] = new double[PopulationFW];

		  double fitness_bestPopulation[] = new double[PopulationFW];
		  
		  
		   for(int i=0; i< this.PopulationFW; i++){
			   ScalingFactor[i] =  RandomGenerator.Randdouble(0, 1);
			   CrossOverRate[i] =  RandomGenerator.Randdouble(0, 1);
		   }
		   
	      for(int i=0; i< PopulationFW; i++){
	          
			  fitness[i] = classficationAccuracy1NN_weights(nominalPopulation, Population[i],trainingDataSet);   // PSOfitness
			  fitness_bestPopulation[i] = fitness[i]; // Initially the same fitness.
		  }
		  
		  
		  //We select the best initial  particle
		  double bestFitness=fitness[0];
		  int bestFitnessIndex=0;
		  for(int i=1; i< PopulationFW;i++){
			  if(fitness[i]>bestFitness){
				  bestFitness = fitness[i];
				  bestFitnessIndex=i;
			  }
			  
		  }
		  
		   for(int j=0;j<PopulationFW;j++){
	         //Now, I establish the index of each prototype.
			  for(int i=0; i<Population.length; ++i)
				  Population[i].setIndex(i);
		   }
		   
  
		   double randj[] = new double[5];
		   
		   
		   for(int iter=0; iter< iterations; iter++){ // Main loop
			      
			   for(int i=0; i<PopulationFW; i++){

				   // Generate randj for j=1 to 5.
				   for(int j=0; j<5; j++){
					   randj[j] = RandomGenerator.Randdouble(0, 1);
				   }
				   

				   if(i==bestFitnessIndex && randj[4] < tau[2]){
					  // System.out.println("SFGSS applied");
					   //SFGSS
					   crossover[i] = SFGSS(Population, i, bestFitnessIndex, CrossOverRate[i],nominalPopulation);
					   
					   
				   }else if(i==bestFitnessIndex &&  tau[2] <= randj[4] && randj[4] < tau[3]){
					   //SFHC
					   //System.out.println("SFHC applied");
					   crossover[i] = SFHC(Population, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i],nominalPopulation);
					   
				   }else {
					   
					   // Fi update
					   
					   if(randj[1] < tau[0]){
						   ScalingFactor[i] = this.Fl + this.Fu*randj[0];
					   }
					   
					   // CRi update
					   
					   if(randj[3] < tau[1]){
						   CrossOverRate[i] = randj[2];
					   }
					   				   
					   // Mutation Operation.
					   
					   mutation[i] = new Prototype();
				   
					  //Mutation:
						
					   mutation[i]  = mutant(Population, i, bestFitnessIndex, ScalingFactor[i]);
					   
					    // Crossver Operation.

					   crossover[i] = new Prototype(Population[i]);
					   
					   for(int j=0; j< Population[i].numberOfInputs(); j++){ // For each part of the solution
						   
						   double randNumber = RandomGenerator.Randdouble(0, 1);
							   
						   if(randNumber<CrossOverRate[i]){
							   crossover[i].setInput(j, mutation[i].getInput(j)); // Overwrite.
						   }
					   }
					   
					   
					   
					   
				   }
				   
	   
				   
				   // Fourth: Selection Operation.
			   

			       fitness[i] =  classficationAccuracy1NN_weights(nominalPopulation,Population[i],trainingDataSet);
   			       double trialVector =  classficationAccuracy1NN_weights(nominalPopulation, crossover[i],trainingDataSet);
				
			  
				  if(trialVector > fitness[i]){
					  Population[i] = new Prototype(crossover[i]);
					  fitness[i] = trialVector;
				  }
				  
				  if(fitness[i]>bestFitness){
					  
					  bestFitness = fitness[i];
					  bestFitnessIndex=i;
					  System.out.println("FITNESSFW= "+ bestFitness);
				  }
				  
				  
			   }

			   
		   }
		  
		  
		   
		   System.out.println("Best weightings");

		   for(int i=0; i<Population[bestFitnessIndex].numberOfInputs(); i++ ){
			   System.out.print(Population[bestFitnessIndex].getInput(i) + "  ");
		   }
		
	  
	  
	  
	  return Population[bestFitnessIndex].getInputs();
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
   * Generate a reduced prototype set by the stratPGFWGenerator method.
   * @return Reduced set by stratPGFWGenerator's method.
 * @throws Exception 
   */
  

  
 public Pair<PrototypeSet, PrototypeSet> applyAlgorithm() throws Exception 
  {
	  System.out.print("\nThe stratification FW process  is starting...\n Computing...\n");
	  
	  this.stratos = Math.round(trainingDataSet.size()/this.StrataSize);
	
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
	  double weights[][] = new double[this.stratos][];
	  
	  for(int i=0; i< this.stratos; i++)
	  {
		  
		  if(this.PGmethod.equalsIgnoreCase("IPADECSFW")){
			  System.out.println("Executing Strata : "+i);
			  
			  IPADECSFWGenerator algorithm = new IPADECSFWGenerator(strata[i], 1, this.MAX_ITER, 10, 50, this.iterSFGSS, this.iterSFHC, this.Fl, this.Fu, 0.1,0.1,  0.03, 0.07, this.Strategy, this.iterFWinitial, this.iterFW, this.PopulationFW);
			  
			  Pair<PrototypeSet, double[]> sol = algorithm.reduceSetFeatures();
			  
			  reduced[i] = new PrototypeSet(sol.first().clone());
			  weights[i] = sol.second().clone();
			  
			  
			  System.out.println("Finish Strata : "+i);
			  
			  algorithm = null;
			  System.gc();
			  
		  }else if (this.PGmethod.equalsIgnoreCase("SSMAPGFW")){
			  
			  System.out.println("Executing Strata : "+i);
			  HandlerSSMAPGFW algorithm = new HandlerSSMAPGFW();
			  
				//  strata[i].print();
				  
			   reduced[i] = new PrototypeSet(algorithm.ejecutar(strata[i].toInstanceSet(), strata[i]));
			   weights[i] = algorithm.getPesos().clone(); 
				   
			 System.out.println("Finish Strata : "+i);
				  
			  algorithm = null;
			  System.gc();
		  }
		  
		  
	  }	  
	  
	  
	  PrototypeSet training = new PrototypeSet(this.trainingDataSet.clone());
	  PrototypeSet test = new PrototypeSet(this.testDataSet.clone());
	  
	  int traPrediction[] = new int[training.size()];
	  int tstPrediction[] = new int[test.size()];
	  int aciertoTrs = 0;
	  int aciertoTst = 0;
	  
	  // combining solutions
	  PrototypeSet solucion = new PrototypeSet();
	  solucion = new PrototypeSet();
	  for(int i=0; i< this.stratos; i++)
	  {
		  solucion.add(reduced[i].clone());
	  }


      
	  
	  double Weights[] = new double[trainingDataSet.get(0).numberOfInputs()]; // weights vector for the FEATURE SPACE!
	  
	  if(this.JoinProcedure.equalsIgnoreCase("JoinOptimize")){
		  
		  // Optimize with DE:
		  Arrays.fill(Weights, 1.0); // initially 1.0
		  System.out.println("Optimizing new weights!");
		  Weights= FeatureWeighting(solucion,true, Weights).clone();
		  
		  // Classify training and test.
		  Prototype Pesos = new Prototype(Weights.clone(), null);
		  
		  PrototypeSet nominalPopulation = new PrototypeSet();
	      nominalPopulation.formatear(solucion);
		  training = new PrototypeSet(NN_weights(nominalPopulation, Pesos, trainingDataSet));
		  test = new PrototypeSet(NN_weights(nominalPopulation, Pesos, testDataSet));
	  
		  
	  }else if(this.JoinProcedure.equalsIgnoreCase("Join")){
		  // Empty at the moment..
		  
	  }else if(this.JoinProcedure.equalsIgnoreCase("VotingRule")){ // Two kinds of Voting Rule: Majority or Kernel 1/(d*d)
		  

		  for(int i=0; i<training.size(); i++){
			  
			  int [] deterClass = new int [this.numberOfClass];
			  double [] distClass = new double [this.numberOfClass];

			  
			  Arrays.fill(distClass, 0);
			  Arrays.fill(deterClass, 0);
			  
			  for(int j=0; j< strata.length; j++){
				  
				  Prototype Pesos = new Prototype(weights[j].clone(), null);
				  Prototype near = _1nnWeighted(training.get(i), Pesos, strata[j]);

				  double dist = 0; //Recalculating distances.
		          for (int m = 0; m < near.numberOfInputs(); m++)
		          {
		        	  dist+= ((training.get(i).getInput(j) - near.getInput(j)) * (training.get(i).getInput(j)  - near.getInput(j)))*Pesos.getInput(j);
		          }
		          
		          distClass[(int)near.getOutput(0)]+= 1/(dist*dist); // Kernel Function
				  deterClass[(int)near.getOutput(0)]++; // increasing the class counter
			  }
			  
			// Majority or  Kernel
			  
			 double maximum = -1;
			 int maxIndex = 0;
			    
			  if(this.Voting.equalsIgnoreCase("Majority")){

			    for (int j = 0; j < this.numberOfClass; j++) {
			      if ((j == 0) || (deterClass[j] > maximum) || (deterClass[j] == maximum && RandomGenerator.Rand()<0.5)) {  // In case of tie..Randomly.
			    	  maxIndex = j;
			    	  maximum = deterClass[j];
			      }
			    }
		    
			  }else if(this.Voting.equalsIgnoreCase("Kernel")){
			
				  for (int j = 0; j < this.numberOfClass; j++) {
					  if ((j == 0) || (distClass[j] > maximum) || (distClass[j] == maximum && RandomGenerator.Rand()<0.5)) {  // In case of tie..Randomly.
				    	  maxIndex = j;
				    	  maximum = deterClass[j];
				      }
				  }

				  
			  }
			  
			  // Label instance.
			  training.get(i).setFirstOutput(maxIndex); // la mejor clase..
			  
		  }
		  
		  
		  for(int i=0; i<test.size(); i++){
			  
			  int [] deterClass = new int [this.numberOfClass];
			  double [] distClass = new double [this.numberOfClass];

			  
			  Arrays.fill(distClass, 0);
			  Arrays.fill(deterClass, 0);
			  
			  for(int j=0; j< strata.length; j++){
				  
				  Prototype Pesos = new Prototype(weights[j].clone(), null);
				  Prototype near = _1nnWeighted(test.get(i), Pesos, strata[j]);

				  double dist = 0; //Recalculating distances.
		          for (int m = 0; m < near.numberOfInputs(); m++)
		          {
		        	  dist+= ((test.get(i).getInput(j) - near.getInput(j)) * (test.get(i).getInput(j)  - near.getInput(j)))*Pesos.getInput(j);
		          }
		          
		          distClass[(int)near.getOutput(0)]+= 1/(dist*dist); // Kernel Function
				  deterClass[(int)near.getOutput(0)]++; // increasing the class counter
			  }
			  
			// Majority or  Kernel
			  
			 double maximum = -1;
			 int maxIndex = 0;
			    
			  if(this.Voting.equalsIgnoreCase("Majority")){

			    for (int j = 0; j < this.numberOfClass; j++) {
			      if ((j == 0) || (deterClass[j] > maximum) || (deterClass[j] == maximum && RandomGenerator.Rand()<0.5)) {  // In case of tie..Randomly.
			    	  maxIndex = j;
			    	  maximum = deterClass[j];
			      }
			    }
		    
			  }else if(this.Voting.equalsIgnoreCase("Kernel")){
			
				  for (int j = 0; j < this.numberOfClass; j++) {
					  if ((j == 0) || (distClass[j] > maximum) || (distClass[j] == maximum && RandomGenerator.Rand()<0.5)) {  // In case of tie..Randomly.
				    	  maxIndex = j;
				    	  maximum = deterClass[j];
				      }
				  }

				  
			  }
			  
			  // Label instance.
			  test.get(i).setFirstOutput(maxIndex); // la mejor clase..
			  
		  }
		  
		  
		  
	  }
	  
	  //Combining the solution: Only joined.
	  /*

	  for(int i=0; i< this.stratos; i++)
	  {
		  solucion.add(reduced[i].clone());
	  }

	  */
	  
	  
	  //solucion.print();

	  
	  
	  
	  // contabilizando
	  for(int i=0; i<training.size(); i++){
   
		    // maxIndex is the class label.		  
		    if(training.get(i).getOutput(0) == trainingDataSet.get(i).getOutput(0)){
				  aciertoTrs++;
		    }

	   }
	  
	  
	  for(int i=0; i<test.size(); i++){
		   
		    // maxIndex is the class label.		  
		    if(test.get(i).getOutput(0) == testDataSet.get(i).getOutput(0)){
				  aciertoTst++;
		    }

	   }
	  
	  
	  
	  System.out.println("% de acierto TRS = "+ (aciertoTrs*100.)/trainingDataSet.size());
	  System.out.println("% de acierto TST = "+ (aciertoTst*100.)/testDataSet.size());
	  System.out.println("Reduction %, result set = "+((trainingDataSet.size()-solucion.size())*100.)/trainingDataSet.size()+ "\n");
	  
	  return new Pair<PrototypeSet,PrototypeSet>(training,test);
  }
  
  /**
   * General main for all the prototoype generators
   * @param args Arguments of the main function.
 * @throws Exception 
   */
  public static void main(String[] args) throws Exception
  {
      Parameters.setUse("stratPGFW", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
      Parameters.assertBasicArgs(args);
      
      PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
      PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
      
      
      long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
      stratPGFWGenerator.setSeed(seed);
      
      int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
      int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
      int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
      int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);

      
      //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
     //System.out.print(" swarm ="+swarm+"\n");
      
      
  	//resultingSet.save(args[1]);
      //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
    
  }

}
