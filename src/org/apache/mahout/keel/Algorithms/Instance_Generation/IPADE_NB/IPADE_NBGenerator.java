
/*
	IPADE_NB.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.

*/

package  org.apache.mahout.keel.Algorithms.Instance_Generation.IPADE_NB;

import java.util.ArrayList;
import java.util.Arrays;


import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.pg.data.Data;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.C45.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSMO;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Distance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Parameters;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.RandomGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;



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
public class IPADE_NBGenerator extends PrototypeGenerator {

  /*Own parameters of the algorithm*/
  
  // We need the variable K to use with k-NN rule
  private int k;
 
  private int PopulationSize; 
  private int ParticleSize;
  private int MaxIter; 
  private int iterBasicDE;
  private double ScalingFactor;
  private double CrossOverRate;
  private double undersampling;
  private int Strategy;
  private String CrossoverType; // Binomial, Exponential, Arithmetic
  private double tau[] = new double[4];
  protected int numberOfClass;
  protected int numberOfPrototypes;  // Particle size is the percentage
  /** Parameters of the initial reduction process. */
  private String[] paramsOfInitialReducction = null;

  private int iterSFGSS;
  private int iterSFHC;
  
  private String classifier;
  private boolean addRand;
  protected int positiveClass;
  Context context;

  
  /**
   * Build a new IPADE_NBGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param perc Reduction percentage of the prototype set.
   */
  
  public IPADE_NBGenerator(PrototypeSet _trainingDataSet, int neigbors,int poblacion, int perc, int iteraciones, double F, double CR, int strg)
  {
      super(_trainingDataSet);
      algorithmName="IPADE_NB";
      
      this.k = neigbors;
      this.PopulationSize = poblacion;
      this.ParticleSize = perc;
      this.MaxIter = iteraciones;
      this.numberOfPrototypes = getSetSizeFromPercentage(perc);
      
      this.ScalingFactor = F;
      this.CrossOverRate = CR;
      this.Strategy = strg;
      
  }
  
  public IPADE_NBGenerator(Context context, Data _trainingDataSet, int neigbors,int iterDE, int iterSFGS, int iterSFHCE, double F, double CR, double tau1, double tau2, String classifier,String random, double under)
  {
	  	  
      super(new PrototypeSet(_trainingDataSet));
      
      //trainingDataSet.print();
      
      this.context=context;
      algorithmName="IPADE_NB";
      this.k =  neigbors;
      this.iterBasicDE = iterDE;//*trainingDataSet.get(0).numberOfInputs(); //NC*1000
      this.iterSFGSS =  iterSFGS;
      this.iterSFHC = iterSFHCE;
      this.ScalingFactor = F;
      this.CrossOverRate = CR;
      this.tau[0] =  tau1;
      this.tau[1] =  tau2;
      
      this.classifier = classifier;
      
      String aleatorio = random;
      
      if(aleatorio.equalsIgnoreCase("true")){
    	  this.addRand = true;
      }else{
    	  this.addRand = false;
      }
      
      
      this.undersampling =  under;
      
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      
	  if(this.numberOfClass == 2){ // si es un problema binario (.. si es para nobalanceados.)
		  double min= Double.MAX_VALUE;
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  if(trainingDataSet.getFromClass(i).size()<min){
				  min=trainingDataSet.getFromClass(i).size();
				  this.positiveClass= i;
			  }
		  }
		  
		  
	  }
	  
      System.out.print("\nIsaac dice:  " + k + " Classifier= "+this.classifier+ " Particle=  "+ ParticleSize + " Maxiter= "+ MaxIter+" CR=  "+this.CrossOverRate+ " CrossverType = "+ this.CrossoverType+"\n");

	  
  }
  


  /**
   * Build a new IPADE_NBGenerator Algorithm
   * @param t Original prototype set to be reduced.
   * @param params Parameters of the algorithm (only % of reduced set).
   */
  public IPADE_NBGenerator(PrototypeSet t, Parameters parameters)
  {
      super(t, parameters);
      algorithmName="IPADE_NB";
      this.k =  parameters.getNextAsInt();
      this.iterBasicDE =  parameters.getNextAsInt();//*trainingDataSet.get(0).numberOfInputs(); //NC*1000
      this.iterSFGSS =  parameters.getNextAsInt();
      this.iterSFHC =  parameters.getNextAsInt();
      this.ScalingFactor = parameters.getNextAsDouble();
      this.CrossOverRate = parameters.getNextAsDouble();
      this.tau[0] =  parameters.getNextAsDouble();
      this.tau[1] =  parameters.getNextAsDouble();
      
      this.classifier = parameters.getNextAsString();
      
      String aleatorio = parameters.getNextAsString();
      
      if(aleatorio.equalsIgnoreCase("true")){
    	  this.addRand = true;
      }else{
    	  this.addRand = false;
      }
      
      
      this.undersampling =  parameters.getNextAsDouble();
      
      this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
      
	  if(this.numberOfClass == 2){ // si es un problema binario (.. si es para nobalanceados.)
		  double min= Double.MAX_VALUE;
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  if(trainingDataSet.getFromClass(i).size()<min){
				  min=trainingDataSet.getFromClass(i).size();
				  this.positiveClass= i;
			  }
		  }
		  
		  
	  }
	  
	  
      System.out.print("\nIsaac dice:  " + k + " Classifier= "+this.classifier+ " Particle=  "+ ParticleSize + " Maxiter= "+ MaxIter+" CR=  "+this.CrossOverRate+ " CrossverType = "+ this.CrossoverType+"\n");
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
       FitnessFi =  GeometricMean(nominalPopulation, trainingDataSet);//accuracy(nominalPopulation,trainingDataSet);
	   
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
  
  

  public int[] classify(PrototypeSet training, PrototypeSet test)
  {
	int predicho[] = new int[test.size()];
	
	
	if(this.classifier.equals("NN")){
	
	  int i=0;
	
	  if(training.size()>this.k){
	      predicho = KNN.classify2(training, test, this.k).clone();
	  }else{
	  
	      for(Prototype p : test)
	      {
	          Prototype nearestNeighbor = KNN._1nn(p, training);          
	    	      	  
	          predicho[i] = (int) nearestNeighbor.getOutput(0);
	                   
	          i++;
	      }
	  }
		
	}else if(this.classifier.equals("C45")){
		C45 c45;
		  
		try {
			 /// training.save("train1.dat");
			 // test.save("test1.dat");
			//  c45 = new C45("train1.dat", "test1.dat");
			c45 = new C45(training.toInstanceSet(), test.toInstanceSet());
		    predicho = c45.getPredictions().clone();   
		    
		    c45 = null;
		    System.gc();
		    
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}      // C4.5 called
    
	}else if(this.classifier.equals("SMO")){
	    HandlerSMO SMO;
	    
		try {
			
			 // training.save("train1.dat");
			  //test.save("test1.dat");
			  //SMO = new HandlerSMO(this.numberOfClass, test.size(),String.valueOf(this.SEED));
			  SMO = new HandlerSMO(training.toInstanceSet(), test.toInstanceSet(),this.numberOfClass,String.valueOf(this.SEED));
			
			  //SMO.generateFiles();
			  
  			  
  		      predicho = SMO.getPredictions(0).clone();    
  		     
  		    SMO = null;
		    System.gc(); 
		     
		      /*
		      for(int i=0; i<test.size(); i++){
		    	  System.out.print(predicho[i]+", ");
		      }
		      System.out.println(" ");
		      */
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}      // SMO
	      
	
	}
    
      return predicho;
  }
  
  //specificity
  public double TPrate(PrototypeSet test, int [] predicho){

	  
	  double tp=0.0,fn=0.0;
	  
	  for(int i=0; i<test.size();i++){
		  
		  if(test.get(i).getOutput(0)==predicho[i] && predicho[i]==this.positiveClass){ // esto es un tp
			  tp++;			  
		  }else if(test.get(i).getOutput(0)!=predicho[i] && test.get(i).getOutput(0)==this.positiveClass){ // es un false
			  fn++;
		  }
	  }
	  
	  return ((1.*tp)/((tp+fn)*1.));
	  
  }
  
  // Specificity
  public double TNrate(PrototypeSet test, int [] predicho){

	  
	  double tn=0.0,fp=0.0;
	  
	  for(int i=0; i<test.size();i++){
		  
		  if(test.get(i).getOutput(0)==predicho[i] && predicho[i]!=this.positiveClass){ // esto es un TN
			  tn++;			  
		  }else if(test.get(i).getOutput(0)!=predicho[i] && predicho[i]==this.positiveClass){ // es un false positve
			  fp++;
		  }
	  }
	  
	  return ((1.*tn)/((tn+fp)*1.));
	  
  }
  
  
  public double FPrate(PrototypeSet test, int [] predicho){

	  
	  double fp=0.0,tn=0.0;
	  
	  for(int i=0; i<test.size();i++){
		  
		  if(test.get(i).getOutput(0)!=predicho[i] && predicho[i]==this.positiveClass){ // esto es un fp
			  fp++;			  
		  }else if(test.get(i).getOutput(0)==predicho[i] && test.get(i).getOutput(0)!=this.positiveClass){ // es un tn
			  tn++;
		  }
	  }
	  
	  return ((1.*fp)/((fp+tn)*1.));
	  
  }
  
  
  public double AUC(PrototypeSet train, PrototypeSet test){
	  double AUC;
	  
	  int []pre = classify(train,test);
	  
	  double tprate= this.TPrate(test, pre);
	  double fprate= this.FPrate(test, pre);
	  
     AUC = (1.0+tprate-fprate)/2.0;
	  
	  
	 // System.out.println("AUC = "+(1.0+tprate-fprate)/2.0);;
	  
	  return AUC;
	  
  }
  
  public double GeometricMean(PrototypeSet train, PrototypeSet test){
	  double GeometricMean;
	  
	  int []pre = classify(train,test);
	  
	  double tprate= this.TPrate(test, pre);
	  double tnrate= this.TNrate(test, pre);
	  
	  GeometricMean = tprate*tnrate;
	  
	  
	 // System.out.println("AUC = "+(1.0+tprate-fprate)/2.0);;
	  
	  return GeometricMean;
	  
  }
  
  /**
   * 
   * @return
   */
  public PrototypeSet basicDE(PrototypeSet myTrain){ 
	 
	  double fitness;
  	  Prototype r1,r2,r3, resta, producto, resta2, producto2;
  	  
      Prototype crossover;
	  
      PrototypeSet nominalPopulation;
	   nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(myTrain);
      
	  fitness = GeometricMean(nominalPopulation, trainingDataSet);// accuracy(nominalPopulation,trainingDataSet);
	  
	  System.out.println("Initial Optim: Fitness "+ fitness);	

	  
   double randj[] = new double[5];
	   
	  // Generate randj for j=1 to 5.
	   for(int j=0; j<5; j++){
		   randj[j] = RandomGenerator.Randdouble(0, 1);
		}
			   
		  for(int i=0; i< this.iterBasicDE; i++){
			  
			  context.progress();

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
					        		 Perturbance.setInput(k, myTrain.get(j).getInput(k)+RandomGenerator.Randdouble(-0.01*l, 0.01*l));
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
		    	
		       double trialFitness = GeometricMean(nominalPopulation, trainingDataSet);
			  //double trialFitness =accuracy(nominalPopulation,trainingDataSet);
				
			  if(trialFitness > fitness){
				  System.out.println("BASICDE: update fitness: "+ trialFitness);
				  fitness = trialFitness;
				  myTrain = new PrototypeSet(modificados.clone());
			  } 

		  
	   
	  }

		  

		  
		  
	  return myTrain;
  }
  
    
  
  
  
 /**
  * Generate a reduced prototype set by the IPADE_NBGenerator method.
  * @return Reduced set by IPADE_NBGenerator's method.
  */
 
 
 public PrototypeSet reduceSetNB()
 {
	  System.out.print("\nThe algorithm  IPADE is starting...\n Computing...\n");
	  
      System.out.print("\nIsaac dice:  " + k + " Classifier= "+this.classifier+ " basicDE=  "+ this.iterBasicDE + " iterSFGSS= "+ this.iterSFGSS+" CR=  "+this.CrossOverRate+ " CrossverType = "+ this.CrossoverType+"\n");

      
	  this.Strategy = 3;
	  
	  PrototypeGenerator.classifier = this.classifier; // IMPORTANT TO DO A GOOD FINAL CLASSIFICATION!
	  PrototypeGenerator.kNearest = this.k;
	  
	  PrototypeSet solucion = new PrototypeSet();
	  
	  PrototypeSet Clases [] = new PrototypeSet[this.numberOfClass];
	  double fitnessClass[] = new double[this.numberOfClass];
	  PrototypeSet nominalPopulation;
	  
	  
	  // Special initialization for C4.5
	  
	  if(this.classifier.equalsIgnoreCase("C45")){

	      C45 c45 = null;
	      boolean [] selectedInstances;

	
	      try {
	          c45 = new C45(trainingDataSet.toInstanceSet(), trainingDataSet.toInstanceSet());
	      } catch (Exception e) {
	          System.err.println("Error during the building of the tree");
	          e.printStackTrace();
	          System.exit(-1);
	      }
	      selectedInstances = c45.selectedTrainingInstances();
	      
	      for(int j=0; j< selectedInstances.length; j++){
	    	//  System.out.print( selectedInstances[j]+",");
	    	  if(selectedInstances[j])
	    		  solucion.add(trainingDataSet.get(j));
		  }
		  
	    //  System.out.println("Procesado el c45");
	      for(int i=0; i<this.numberOfClass; i++){
	    	  Clases[i] = new PrototypeSet(solucion.getFromClass(i).clone());
			  
			  
			  if(Clases[i].size()<1){ // in this extreme case...
				  int numberOfPrototypes;
				  
				  if(i==this.positiveClass) 
					  numberOfPrototypes = (int)Math.round(Clases[i].size()*this.undersampling);// 80%
				  else
					  numberOfPrototypes = (int)Math.round(Clases[i].size()*(1-this.undersampling));// 20%
					  
				  
				  
				  System.out.println("numberOfPrototypes: "+numberOfPrototypes);
				  
				  if(numberOfPrototypes < 1){numberOfPrototypes=2;}
				  
				  for(int j=0; j< numberOfPrototypes; j++){
					  solucion.add(trainingDataSet.getFromClass(i).getRandom());
				  }
				  
			  }
			  
			  System.out.println("Size ->"+solucion.getFromClass(i).size());

	      }
		  
	  }else{
	
		  for(int i=0; i<this.numberOfClass; i++){
			 
			  
			  if(trainingDataSet.getFromClass(i).size() >0){ 
				  Clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i).clone());
				  
				  System.out.println("Size ->"+Clases[i].size());
				  
				  
			
				  
	/*
				  
					  /// Dep. de la clase...
					  
					  int numberOfPrototypes;
					  
					  if(i==this.positiveClass) 
						  numberOfPrototypes = (int)Math.round(Clases[i].size()*this.undersampling);// 80%
					  else
						  numberOfPrototypes = (int)Math.round(Clases[i].size()*(1-this.undersampling));// 20%
						  
					  
					  
					  System.out.println("numberOfPrototypes: "+numberOfPrototypes);
					  
					  if(numberOfPrototypes < 1){numberOfPrototypes=1;}
					  
					  for(int j=0; j< numberOfPrototypes; j++){
						  solucion.add(Clases[i].getRandom());
					  }
					  
		*/		  
				  
				
				  
				 //			  Inicialición habitual:
				  if(this.classifier.equalsIgnoreCase("NN")){ // para nearest neighbor el centroide...
					  Prototype centroid = Clases[i].avg();
					  //centroid.print();
					  solucion.add(centroid); // Centroide
				  
				  }else{
				  
				   					  //sino un metodo de IS muy reductivo para esa clase....
					  
					  // Al menos un subconjunto más grande aleatorio... 
					  
					  
					  
					  int numberOfPrototypes = (int)Math.round(Clases[i].size()*0.5);// 20%
					  if(numberOfPrototypes < 1){numberOfPrototypes=1;}
					  
					  for(int j=0; j< numberOfPrototypes; j++){
						  solucion.add(Clases[i].getRandom());
					  }
					  
				  }
				  
				  
				  
			  } //end if.
			 
		  }  //end for
	  } //end else.
	  
	  solucion.print();
	  
	  //int iteraciones = this.iterBasicDE;
	  
	//  this.iterBasicDE= 100;
	  solucion=basicDE(solucion); //Initial Optimization
	 // solucion.print();
	 // this.iterBasicDE= iteraciones;
	  
	  double Fitness= GeometricMean(solucion, trainingDataSet);//accuracy(solucion,trainingDataSet);
	//  System.out.println("Initial Global Fitness = "+ Fitness);
	  
	  boolean claseMarcada[] = new boolean[this.numberOfClass];
	  boolean fin[] = new boolean[this.numberOfClass];
	  Arrays.fill(claseMarcada, false);
	  Arrays.fill(fin, true);
	  
	  int iterOptimizada[] = new int [this.numberOfClass];
	  Arrays.fill(iterOptimizada, 1);
	  
	  int contOptimizedPositive[] = new int[this.numberOfClass];
	  Arrays.fill(contOptimizedPositive, 0);
	  int iter=0;
	  
	  PrototypeSet solRescatada = null;
	  while(!Arrays.equals(claseMarcada, fin)){	  
		  
		//  System.out.println("iter "+ iter+ " fitness= "+Fitness);
		 // System.out.println("iter "+ iter+ " Red= "+((trainingDataSet.size()-solucion.size())*100.)/trainingDataSet.size());
		  
		  
		  double minFitness= Double.MAX_VALUE;
		  int objetivo= -1;
		  
		  for(int j=0; j<this.numberOfClass; j++){
			  if(trainingDataSet.getFromClass(j).size()>1){
				  
				  
				  fitnessClass[j]= accuracy(solucion,trainingDataSet.getFromClass(j)); // AUC(solucion, trainingDataSet.getFromClass(j));//
				   
				  System.out.println("Fitness class["+j+"]= " +fitnessClass[j]);
				  
				  
				  if(fitnessClass[j] < minFitness && !claseMarcada[j]){
					  minFitness = fitnessClass[j];
					  objetivo = j;
				  }
				  
				 /* if(fitnessClass[j] == 100){
					  claseMarcada[j] = true;
				  }*/
			  }else{
				  claseMarcada[j] = true;
			  } 
		   }
		 
			  
		  System.out.println("Objective =" + objetivo + ", Clase minoritaria ="+this.positiveClass);
		  PrototypeSet tester;
		 
		  
		 
			  if(!claseMarcada[objetivo]){
				  PrototypeSet solucion2;
	
				  
				  if(objetivo==this.positiveClass &&  contOptimizedPositive[objetivo]> 0){ // lo que persigo es, si no he mejorado, me faltan iteraciones, solo se le permite a la clase minoritaria
					  solucion2 = new PrototypeSet(solRescatada.clone());
					  
				  }else{
 
					  solucion2 = new PrototypeSet(solucion.clone());
					  
					  Prototype Addition =null;
					  
					  if(this.addRand || objetivo!=this.positiveClass){
					  	  Addition= trainingDataSet.getFromClass(objetivo).getRandom();
					  
					  }else{ // Quiero coger el más lejano de trainingDataSet.getFromClass(objetivo)  a todo lo que yo tengo ahora mismo.
						  PrototypeSet delaClase = trainingDataSet.getFromClass(objetivo);
						  
						  int lejano =0;
						  double distLejano = Double.MAX_VALUE;
						  for(int z=0; z< delaClase.size();z++){
							  
							  double dist = 0;
							  for(int h=0; h<solucion2.size(); h++){
								  
								  double diz= Distance.absoluteDistance(solucion2.get(h), delaClase.get(z));
								  if(diz!=0){
									  dist+=diz;
								  }else{
									  dist+=Double.MAX_VALUE;
								  }
							  }
							  
							  if(dist<distLejano && dist !=0){
								  distLejano = dist;
								  lejano=z;
							  }
						  }
						  
						  System.out.println("Lejano "+ lejano);
						  Addition= trainingDataSet.getFromClass(objetivo).get(lejano);
					  }
					  
					  solucion2.add(new Prototype(Addition)); // A�ado uno Y pruebo a optimizar.
				  }
				 // solucion2.add(trainingDataSet.farthestTo(solucion.getFromClass(objetivo).getRandom())); 
				  
				  tester = basicDE(solucion2).clone();
			  		  
				//   nominalPopulation = new PrototypeSet();
			     //  nominalPopulation.formatear(solucion);
			  //     Fitness=  GeometricMean(nominalPopulation, trainingDataSet);// accuracy(nominalPopulation,trainingDataSet);
			       
			       nominalPopulation = new PrototypeSet();
			       nominalPopulation.formatear(tester);
				   double trialFitness=   GeometricMean(nominalPopulation, trainingDataSet);//accuracy(nominalPopulation,trainingDataSet);
				  
				  
				  System.out.println("Trial fitnss= " + trialFitness);
				  if(trialFitness > Fitness ){ // le ponemos el >=, aquí no hay miedo por la reducción.
					    
					  iterOptimizada[objetivo]++;
					  solucion = new PrototypeSet(tester.clone());
					  Fitness = trialFitness;
					  contOptimizedPositive[objetivo]= 0; // reinicializo la cuenta.
					     System.out.println("añado de la clase ->" +objetivo);
				  }else if(trialFitness==Fitness && objetivo==this.positiveClass && iterOptimizada[objetivo]<(trainingDataSet.getFromClass(objetivo).size()*2)){
					  
					  iterOptimizada[objetivo]++;
				      solucion = new PrototypeSet(tester.clone());
				      Fitness = trialFitness;
				  
				      System.out.println("añado de la clase AQUI ->" +objetivo);
			     }else{
			    	 if(objetivo==this.positiveClass){
			    		 
			    		 solRescatada = new PrototypeSet(tester.clone());
					  contOptimizedPositive[objetivo]++; // contar que la clase positiva no ha sido optimizada. 
					  
					  System.out.println("cont ->" +contOptimizedPositive[objetivo]);
					  if(contOptimizedPositive[objetivo]>=10){ // si ya van 5 veces que no consigo mejorarla, bueno, pues paramos.
						  claseMarcada[objetivo] = true;
					  }
			    	 }else{
			    		 claseMarcada[objetivo] = true;
			    	 }
				  }
				  
			  }
		  
			  //Fitness= accuracy(solucion,trainingDataSet);
			  System.out.println("Fitness = "+ Fitness);
		  
			  iter++;
		  
		  
	  }
	  
	  //solucion.print();
	  nominalPopulation = new PrototypeSet();
     nominalPopulation.formatear(solucion);
	  double trialFitness=  GeometricMean(nominalPopulation, trainingDataSet);// accuracy(nominalPopulation,trainingDataSet);
	  
	  System.out.println("Final Fitness = "+ trialFitness);
	  System.out.println("Reduction %, result set = "+((trainingDataSet.size()-solucion.size())*100.)/trainingDataSet.size()+ "\n");
	  
	  for(int i=0; i<this.numberOfClass; i++){
		  System.out.println("Size -> "+solucion.getFromClass(i).size());
	  }
	  
	//  solucion.print();

	//  ParametersC45.prune = true;
	//  ParametersC45.itemsetsPerLeaf =2 ;

    return nominalPopulation;
 }
 
 /**
  * General main for all the prototoype generators
  * @param args Arguments of the main function.
* @throws Exception 
  */
 public static void main(String[] args) throws Exception
 {
     Parameters.setUse("IPADE_NB", "<seed> <Number of neighbors>\n<Swarm size>\n<Particle Size>\n<MaxIter>\n<DistanceFunction>");        
     Parameters.assertBasicArgs(args);
     
     PrototypeSet training = PrototypeGenerationAlgorithm.readPrototypeSet(args[0]);
     PrototypeSet test = PrototypeGenerationAlgorithm.readPrototypeSet(args[1]);
     
     
     long seed = Parameters.assertExtendedArgAsInt(args,2,"seed",0,Long.MAX_VALUE);
     IPADE_NBGenerator.setSeed(seed);
     
     int k = Parameters.assertExtendedArgAsInt(args,3,"number of neighbors", 1, Integer.MAX_VALUE);
     int swarm = Parameters.assertExtendedArgAsInt(args,4,"swarm size", 1, Integer.MAX_VALUE);
     int particle = Parameters.assertExtendedArgAsInt(args,5,"particle size", 1, Integer.MAX_VALUE);
     int iter = Parameters.assertExtendedArgAsInt(args,6,"max iter", 1, Integer.MAX_VALUE);

     
     //String[] parametersOfInitialReduction = Arrays.copyOfRange(args, 4, args.length);
    //System.out.print(" swarm ="+swarm+"\n");
     
     
     IPADE_NBGenerator generator = new IPADE_NBGenerator(training, k,swarm,particle,iter, 0.5,0.5,1);
     
 	  
     PrototypeSet resultingSet = generator.execute();
     
 	//resultingSet.save(args[1]);
     //int accuracyKNN = KNN.classficationAccuracy(resultingSet, test, k);
     int accuracy1NN = KNN.classficationAccuracy(resultingSet, test);
     generator.showResultsOfAccuracy(Parameters.getFileName(), accuracy1NN, test);
 }

}
