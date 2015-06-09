
/*
	CoDE2.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  6-2-2011
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package org.apache.mahout.keel.Algorithms.Instance_Generation.CoDE2;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Chen.ChenGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.HYB.HYBGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.PSO.PSOGenerator;
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
public class CoDE2Generator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
	
 private int MAX_ITER;
	
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int ParticleSize;
  private int MaxIter; 
  private int Strategy;
  private double Beta;
  private double ScalingFactor;
  private double CrossOverRate;
  
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;

  private double tau[] = new double[4];
  private double Fl, Fu;
  
  private int iterSFGSS;
  private int iterSFHC;
  
  /**
   * Build a new CoDE2Generator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public CoDE2Generator(PrototypeSet _trainingDataSet, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="CoDE2";
      
      this.k = neigbors;
      this.ParticleSize = perc;
      this.MaxIter = iteraciones;
      this.numberOfPrototypes = getSetSizeFromPercentage(perc);
      
  }
  


  /**
   * Build a new CoDE2Generator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public CoDE2Generator(PrototypeSet t, Parameters parameters)
  {
      super(t, parameters);
      algorithmName="CoDE2";
      
      this.k =  parameters.getNextAsInt();
      this.MAX_ITER = parameters.getNextAsInt();

      this.ParticleSize =  parameters.getNextAsInt();
      this.MaxIter =  parameters.getNextAsInt();
      this.iterSFGSS =  parameters.getNextAsInt();
      this.iterSFHC =  parameters.getNextAsInt();
      this.Fl = parameters.getNextAsDouble();
      this.Fu = parameters.getNextAsDouble();
      this.tau[0] =  parameters.getNextAsDouble();
      this.tau[1] =  parameters.getNextAsDouble();
      this.tau[2] =  parameters.getNextAsDouble();
      this.tau[3] =  parameters.getNextAsDouble();
      this.Strategy =  parameters.getNextAsInt();
      this.Beta = parameters.getNextAsDouble(); // 0.5;
      
      
      this.numberOfPrototypes = getSetSizeFromPercentage(ParticleSize);
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      System.out.print("\nIsaac dice:  " + k + " Particle=  "+ ParticleSize + " Maxiter= "+ MaxIter+" tau4=  "+this.tau[3]+ "\n");
      //numberOfPrototypes = getSetSizeFromPercentage(parameters.getNextAsDouble());
  }
  
  
  
  public PrototypeSet mutant(PrototypeSet population, double SFi){
	  
	  
	  PrototypeSet mutant = new PrototypeSet(population.clone());
	  Prototype r1,r2,r3,r4,r5, resta,resta2,producto2, producto, nearest;
	  

	  for(int i=0; i< population.size(); i++){
      
		  //PrototypeSet mismaClase= KNN.getNearestNeighborsWithSameClassAs(population.get(i), population, 5);
		  //PrototypeSet mismaClase= KNN.getNearestNeighborsWithSameClassAs(population.get(i), trainingDataSet, 3);
	      
		   PrototypeSet mismaClase = trainingDataSet.getFromClass(population.get(i).getOutput(0));
		  

	      PrototypeSet aux = new PrototypeSet();
		  
	      if(mismaClase.size() < 5){

	    	  for(int j=mismaClase.size(); j < 5; j++){
	    		Prototype Perturbance = new Prototype(population.get(i));

	    		for(int k=0; k< Perturbance.numberOfInputs(); k++){
	        		 Perturbance.setInput(k, population.get(i).getInput(k)+RandomGenerator.Randdouble(-0.01*j, 0.01*j));
	        	}
	    		aux.add(Perturbance);
	    		
	    	  }
	    	  
	    	  mismaClase.add(aux);
	    	  
	    	  
	      }
	      
	      int lista[] = new int[mismaClase.size()];
	      inic_vector_sin(lista,i);
	      desordenar_vector_sin(lista);
	      
	      
	       r1 = mismaClase.get(lista[0]);
		   r2 =  mismaClase.get(lista[1]);
		   r3 =  mismaClase.get(lista[2]);
		   r4 =  mismaClase.get(lista[3]);
		   r5 =  mismaClase.get(lista[4]);
		   
			switch(this.Strategy){
				case 1:// ViG = Xr1,G + F(Xr2,G - Xr3,G) De rand 1
					resta = r2.sub(r3);
					producto = resta.mul(SFi);
					mutant.set(i, producto.add(r1));
				break;
			

				case 2: //DE rand to nearest 1
					resta = r1.sub(r2);
					nearest = KNN.getNearestNeighborsWithSameClassAs(population.get(i), trainingDataSet, 1).get(0);
					
					resta2 = nearest.sub(population.get(i));
					
					producto = resta.mul(SFi);
					producto2 = resta.mul(SFi);
					
					producto = producto.add(producto2);
					mutant.set(i, (population.get(i)).add(producto));
				
				break;
					
	  		       
				case 3://DE current to rand 1
					resta = r2.sub(r3);
					resta2= r1.sub(population.get(i));
					
					double aleatorio = RandomGenerator.Randdouble(0, 1);
					producto = resta.mul(SFi*aleatorio);
					producto2 = resta2.mul(aleatorio);
					
					producto = producto.add(producto2);
					
					mutant.set(i, producto.add(population.get(i)));
				break;
				
				
				case 4://  De rand 2
					resta = r2.sub(r3);
					resta2= r4.sub(r5);
					
					producto = resta.mul(SFi);
					producto2 = resta2.mul(SFi);
					
					producto = producto.add(producto2);
					
					mutant.set(i, producto.add(r1));
				break;
		
				
				
			}
		  
	  }
		   

	 // System.out.println("********Mutant**********");
	 // mutant.print();
	   
     mutant.applyThresholds();
	
	  return mutant;
  }
  
  

  /**
   * Local Search Fitness Function
   * @param Fi
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   */
  public double lsff(double Fi, double CRi, PrototypeSet population){
	  PrototypeSet resta, producto, mutation;
	  PrototypeSet crossover;
	  double FitnessFi = 0;
	  
	  
	  //Mutation:
	  mutation = new PrototypeSet(population.size());
   	  mutation = mutant(population, Fi);
   	
   	  //Crossover
   	crossover =new PrototypeSet(mutation);
   	  /*crossover =new PrototypeSet(population.clone());
   	  
	   for(int i=0; i< mutation.size(); i++){
			for(int j=0; j< mutation.get(i).numberOfInputs(); j++){
				   double randNumber = RandomGenerator.Randdouble(0, 1);
				   
				   if(randNumber<CRi){
					   Prototype Aux = mutation.get(i);
					   crossover.get(i).setInput(j, Aux.getInput(j)); // Overwrite.
				   }
			   
			}

	   }
	   
	  */ 
	   // Compute fitness
	   PrototypeSet nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(crossover);
       FitnessFi = accuracy(nominalPopulation,trainingDataSet);
	   
   	   return FitnessFi;
  }
  
  
  
  /**
   * SFGSS local Search.
   * @param population
   * @return
   */
  public PrototypeSet SFGSS(PrototypeSet population, double CRi){
	  double a=0.1, b=1;
	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
	  double phi = (1+ Math.sqrt(5))/5;
	  double scaling;
	  PrototypeSet crossover, resta, producto, mutation;
	  
	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
	  
		  fi1 = b - (b-a)/phi;
		  fi2 = a + (b-a)/phi;
		  
		  fitnessFi1 = lsff(fi1, CRi, population);
		  fitnessFi2 = lsff(fi2, CRi,population);
		  
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
	  mutation = new PrototypeSet(population.size());
   	  mutation = mutant(population, scaling);
   	
   	  //Crossover
  	crossover =new PrototypeSet(mutation);
  	
 	  /*crossover =new PrototypeSet(population.clone());
   	  
	   for(int i=0; i< mutation.size(); i++){
			for(int j=0; j< mutation.get(i).numberOfInputs(); j++){
				   double randNumber = RandomGenerator.Randdouble(0, 1);
				   
				   if(randNumber<CRi){
					   Prototype Aux = mutation.get(i);
					   crossover.get(i).setInput(j, Aux.getInput(j)); // Overwrite.
				   }
			   
			}

	   }
	   
	   */
	  
	return crossover;
  }
  
  /**
   * SFHC local search
   * @param xt
   * @param xr
   * @param xs
   * @param actual
   * @param SFi
   * @return
   */
  
  public  PrototypeSet SFHC(PrototypeSet population, double SFi, double CRi){
	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
	  PrototypeSet crossover, resta, producto, mutation;
	  double h= 0.5;
	  
	  
	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
		  		  
		  fitnessFi1 = lsff(SFi-h, CRi, population);
		  fitnessFi2 = lsff(SFi, CRi,  population);
		  fitnessFi3 = lsff(SFi+h, CRi,  population);
		  
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
	  mutation = new PrototypeSet(population.size());
   	  mutation = mutant(population, SFi);
   	
   	  //Crossover
  	crossover =new PrototypeSet(mutation);
  	
 	  /*  crossover =new PrototypeSet(population.clone());
   	  
	   for(int i=0; i< mutation.size(); i++){
			for(int j=0; j< mutation.get(i).numberOfInputs(); j++){
				   double randNumber = RandomGenerator.Randdouble(0, 1);
				   
				   if(randNumber<CRi){
					   Prototype Aux = mutation.get(i);
					   crossover.get(i).setInput(j, Aux.getInput(j)); // Overwrite.
				   }
			   
			}

	   }
	   
	   */
	   
	  
	return crossover;
  
  }
  
  
  
  
  /**
   * 
   * @return
   */
  public PrototypeSet doGeneration(PrototypeSet poblacion, double claseObjetivo, PrototypeSet bestIndividual){ 
	 
	  double fitness;
  	  Prototype r1,r2,r3, resta, producto, resta2, producto2;
  	  
      Prototype crossover;
	  
      PrototypeSet nominalPopulation;
	   nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(poblacion);
      
	  
       fitness = fitnessFunction(nominalPopulation, claseObjetivo, bestIndividual);//accuracy(nominalPopulation,trainingDataSet);
	  // System.out.println("fitness "+ fitness);	

	  PrototypeSet myTrain = new PrototypeSet(poblacion.getFromClass(claseObjetivo).clone()); // I select the class to optimize
       
	  ScalingFactor =  RandomGenerator.Randdouble(0, 1);
	  CrossOverRate =  RandomGenerator.Randdouble(0, 1);
	  
   double randj[] = new double[5];
	   
	  // Generate randj for j=1 to 5.
	   for(int j=0; j<5; j++){
		   randj[j] = RandomGenerator.Randdouble(0, 1);
		}
			   
		  for(int i=0; i< this.MaxIter; i++){
			  
			  
			  PrototypeSet modificados = new PrototypeSet(myTrain);
			  
			  if(i%1000==0){ 
				  if(randj[4] < tau[0]){
					  // System.out.println("SFGSS applied");
					   //SFGSS
					   modificados = SFGSS(myTrain, this.CrossOverRate);
					   
					   
				   }else if(tau[0] <= randj[4] && randj[4] < tau[1]){
					  modificados = SFHC(myTrain, this.ScalingFactor, this.CrossOverRate);
				  }
			  }else{ 
			 
				  ScalingFactor = 0.1+ 0.9*RandomGenerator.Randdouble(0, 1);
				  
				  for(int j=0; j< myTrain.size(); j++){
					  
					   PrototypeSet mismaClase = trainingDataSet.getFromClass(myTrain.get(j).getOutput(0));
						  
	
					      PrototypeSet aux = new PrototypeSet();
						  
					      if(mismaClase.size() < 3){
	
					    	  for(int l=mismaClase.size(); l < 5; l++){
					    		Prototype Perturbance = new Prototype(myTrain.get(j));
	
					    		for(int k=0; k< Perturbance.numberOfInputs(); k++){
					        		 Perturbance.setInput(k, myTrain.get(j).getInput(k)+RandomGenerator.Randdouble(-0.15*l, 0.15*l));
					        	}
					    		aux.add(Perturbance);
					    		
					    	  }
					    	  
					    	  mismaClase.add(aux);
					    	  
					    	  
					      }
					      
				      ArrayList<Integer> indexes =  RandomGenerator.generateDifferentRandomIntegers(0, mismaClase.size()-1);
			           r1 = mismaClase.get(indexes.get(0));
			    	   r2 = mismaClase.get(indexes.get(1));
			    	   r3 = mismaClase.get(indexes.get(2));
			    	   
					  	  	   
			    	   	  
					//DE current to rand 1
						resta = r2.sub(r3);
						resta2= r1.sub(myTrain.get(j));
						
						double aleatorio = RandomGenerator.Randdouble(0, 1);
						producto = resta.mul(this.ScalingFactor*aleatorio);
						producto2 = resta2.mul(aleatorio);
						
						producto = producto.add(producto2);
						
						crossover = producto.add(myTrain.get(j)); // Current
				
						crossover.applyThresholds();
	
						modificados.set(j,crossover);
						  
				  } // End mutation and crossover
			  } //end else

			  
			   nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(modificados);
		    		       
			  double trialFitness =fitnessFunction(nominalPopulation,  claseObjetivo, bestIndividual);
				
			  if(trialFitness > fitness){
				  //System.out.println("Selecting");
				  fitness = trialFitness;
				  myTrain = new PrototypeSet(modificados.clone());
			  } 

		  
	   
	  }

		  

		  
		  
	  return myTrain;
  }
  
  
  /** Main method */
  
  public PrototypeSet reduceSet(){
	  
	  
	  // A population per class.
	  
	  PrototypeSet population[] = new PrototypeSet[this.numberOfClass];
	  double localAcc[] = new double[this.numberOfClass];
	  double fitness[] = new double[this.numberOfClass];
	  
	  
	  
	  // First Stage, Initialization.
	  
	  PrototypeSet Initial=new PrototypeSet(selecRandomSet(numberOfPrototypes,true).clone());
	  PrototypeSet bestSolution;
	  
	  // Aseguro que al menos hay un representante de cada clase.
	  PrototypeSet clases[] = new PrototypeSet [this.numberOfClass];
	  
	  for(int i=0; i< this.numberOfClass; i++){
		  clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i));
	  }
	
	  for(int i=0; i< Initial.size(); i++){
		  for(int j=0; j< this.numberOfClass; j++){
			  if(Initial.getFromClass(j).size() ==0 && clases[j].size()!=0){
				  
				  Initial.add(clases[j].getRandom());
			  }
		  }
	  }
	  
	  bestSolution = new PrototypeSet(Initial); // initially this is the best solution
	  
	  // Calculate initial fitness
	  for(int i=0; i< this.numberOfClass; i++){
		  fitness[i] = fitnessFunction(Initial, i, bestSolution);
	  }


	  //Initial.print();
	  
	  PrototypeSet nominalPopulation = new PrototypeSet();
      nominalPopulation.formatear(Initial);
      
      
      //Initial.print();
	// System.err.println("\n% de acierto en training Nominal " + KNN.classficationAccuracy(nominalPopulation,trainingDataSet,1)*100./trainingDataSet.size() );
  
	  
	  // Co-evolutionary stage
	  
	  int iter =0;
	  
		while(iter<MAX_ITER){
			

			
			for(int i=0; i<this.numberOfClass;i++){
				 // Do generation...population[i][j]
				if(Initial.getFromClass(i).size()>0){
					population[i]= new PrototypeSet(doGeneration(Initial, i, bestSolution));	
				}
			}
			
			//updateCollaborators; // who is the best ?

			PrototypeSet nuevo= new PrototypeSet(population[0].clone());
			
			for(int i=1; i< this.numberOfClass; i++){
				if(Initial.getFromClass(i).size()>0){
					nuevo.add(population[i]);
				}
			}
			
			//nuevo.print();
			
			PrototypeSet Actual = new PrototypeSet();
			
			for(int i=0; i< this.numberOfClass; i++){
						
				if(Initial.getFromClass(i).size()>0){
					double fitnessN= fitnessFunction(nuevo, i, bestSolution);
					  
					 if(fitnessN > fitness[i]){
						  fitness[i] = fitnessN;
						  Actual.add(nuevo.getFromClass(i).clone());
					  }else{
						  Actual.add(Initial.getFromClass(i).clone());
					  }
				  
				}
				
			}
			

			Initial = new PrototypeSet(Actual.clone());
			
			iter++;
		}
	  
		
		// Generate Final reference set.
		

		  nominalPopulation = new PrototypeSet();
           nominalPopulation.formatear(Initial);
           
    
           Initial.print();
    	 System.err.println("\n% de acierto en training Nominal " + KNN.classficationAccuracy(nominalPopulation,trainingDataSet,1)*100./trainingDataSet.size() );
		
    	 System.out.println("Reduction % " + (100.-(nominalPopulation.size()*100.)/trainingDataSet.size()) );
         
		
	  
	  return nominalPopulation;
  }
  
  /**
   * Initial fitness fuction
   * @param population
   * @param claseObjetivo
   * @param particle
   * @return
   */
  
  public double fitnessFunction(PrototypeSet solucion, double claseObjetivo, PrototypeSet bestIndividual){
	  double fitness =0;

	  PrototypeSet newSolution= new PrototypeSet(solucion.getFromClass(claseObjetivo));
	  
	  for(int i=0; i<this.numberOfClass; i++){
		  if(i!= claseObjetivo){
			  newSolution.add(bestIndividual.getFromClass(i));
		  }
	  }
	  
	  fitness = this.Beta*accuracy(newSolution,trainingDataSet)+  (1-this.Beta)*accuracy(newSolution,trainingDataSet.getFromClass(claseObjetivo)); 
	  
	  return fitness;
	  
  }
  
 
  
  /**
   * General main for all the prototoype generators
   * Arguments:
   * 0: Filename with the training data set to be condensed.
   * 1: Filename which contains the test data set.
   * 3: Seed of the random number generator.            Always.
   * **************************
   * 4: .Number of neighbors
   * 5:  Swarm Size
   * 6:  Particle Size
   * 7:  Max Iter
   * 8:  C1
   * 9: c2
   * 10: vmax
   * 11: wstart
   * 12: wend
   * @param args Arguments of the main function.
 * @throws Exception 
   */
  public static void main(String[] args) throws Exception
  {
      Parameters.setUse("CoDE2", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
      Parameters.assertBasicArgs(args);
      
      PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
      PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
      
      
      long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
      CoDE2Generator.setSeed(seed);
      
      int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
      int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
      int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
      int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);
      double c1 = Parameters.assertExtendedArgAsInt(args,7,"c1", 1, Double.MAX_VALUE);
      double c2 =Parameters.assertExtendedArgAsInt(args,8,"c2", 1, Double.MAX_VALUE);
      double vmax =Parameters.assertExtendedArgAsInt(args,9,"vmax", 1, Double.MAX_VALUE);
      double wstart = Parameters.assertExtendedArgAsInt(args,10,"wstart", 1, Double.MAX_VALUE);
      double wend =Parameters.assertExtendedArgAsInt(args,11,"wend", 1, Double.MAX_VALUE);
      
      //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
     //System.out.print(" swarm ="+swarm+"\n");
      
      
      CoDE2Generator generator = new CoDE2Generator(training, k,swarm,particle,iter, 0.5,0.5,1);
      
  	  
      PrototypeSet resultingSet = generator.execute();
      
  	//resultingSet.save(args[1]);
      //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
      int accuracy1NN = KNN.classficationAccuracy(resultingSet, test);
      generator.showResultsOfAccuracy(Parameters.getFileName(), accuracy1NN, test);
  }

}
