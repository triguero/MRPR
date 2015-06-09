//
//  SSMA.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 3-10-2005.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.SSMACoDE;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.*;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;



import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;

import org.core.*;

import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.Arrays;

public class SSMACoDE extends Metodo {

  /*Own parameters of the algorithm*/

  private long semilla;
  private int tamPoblacion;
  private double nEval;
  private double pCross;
  private double pMut;
  private int kNeigh;
  public String Script; // para releer par�metros..
  private PrototypeSet trainingDataSet;
  private PrototypeSet testDataSet;
  private PrototypeGenerator generador;
  //Par�metros DE
  private int k;
  
  private String tipoFitness;
  private int PopulationSize; 
  private int ParticleSize;
  private int MaxIter; 
  private double ScalingFactor;
  private double CrossOverRate;
  private int Strategy;
  private String CrossoverType; // Binomial, Exponential, Arithmetic
  
  private int MAX_ITER;
  private double Beta;
  
  private double tau[] = new double[4];
  private double Fl, Fu;
  
  private int iterSFGSS;
  private int iterSFHC;
  
  protected int numberOfClass;


  protected int numberOfPrototypes;  // Particle size is the percentage
  protected int numberOfStrategies; // number of strategies in the pool
  
  protected int positiveClass;
  
  public SSMACoDE (String ficheroScript) {
	super (ficheroScript);
    
  }


  /**
   * Reads the prototype set from a data file.
   * @param nameOfFile Name of data file to be read.
   * @return PrototypeSet built with the data of the file.
   */
  public static PrototypeSet readPrototypeSet(String nameOfFile)
  {
      Attributes.clearAll();//BUGBUGBUG
      InstanceSet training = new InstanceSet();        
      try
      {
      	//System.out.print("PROBANDO:\n"+nameOfFile);
          training.readSet(nameOfFile, true); 
          training.setAttributesAsNonStatic();
          InstanceAttributes att = training.getAttributeDefinitions();
          Prototype.setAttributesTypes(att);            
      }
      catch(Exception e)
      {
          System.err.println("readPrototypeSet has failed!");
          e.printStackTrace();
      }
      return new PrototypeSet(training);
  }
  
  public void inic_vector_sin(int vector[], int without){

  	for(int i=0; i<vector.length; i++) 
  		if(i!=without)
  			vector[i] = i; // Lo inicializo de 1 a n-1
  }
  
  public void desordenar_vector_sin(int vector[]){
  	int tmp, pos;
  	for(int i=0; i<vector.length-1; i++){
  		pos = Randomize.Randint(0, vector.length-1);
  		tmp = vector[i];
  		vector[i] = vector[pos];
  		vector[pos] = tmp;
  	}
  }
  

  
  public PrototypeSet mutant(PrototypeSet population[], int actual, int mejor, double SFi){
  	  
  	  
  	  PrototypeSet mutant = new PrototypeSet(population.length);
  	  PrototypeSet r1,r2,r3,r4,r5, resta, producto, resta2, producto2, result, producto3, resta3;
  	  
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
  		   		 resta = r2.restar(r3);
  		   		 producto = resta.mulEscalar(SFi);
  		   		 mutant = producto.sumar(r1);
  		   	    break;
  			   
  		   	   case 2: // Vig = Xbest,G + F(Xr2,G - Xr3,G)  De best 1
  			   		 resta = r2.restar(r3);
  			   		 producto = resta.mulEscalar(SFi);
  			   		 mutant = population[mejor].sumar(producto);
  			   break;
  			   
  		   	   case 3: // Vig = ... De rand to best 1
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = population[mejor].restar(population[actual]);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = population[actual].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   		 			   		 
  			   break;
  			   
  		   	   case 4: // DE best 2
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = r3.restar(r4);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = population[mejor].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   break;
  			  
  		   	   case 5: //DE rand 2
  		   		   resta = r2.restar(r3); 
  		   		   resta2 = r4.restar(r5);
  		   		 			   		 
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   		
  			   	   result = r1.sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   	   
    		       break;
    		       
  		   	   case 6: //DE rand to best 2
  		   		   resta = r1.restar(r2); 
  		   		   resta2 = r3.restar(r4);
  		   		   resta3 = population[mejor].restar(population[actual]);
  		   		   
  			   	   producto = resta.mulEscalar(SFi);
  			   	   producto2 = resta2.mulEscalar(SFi);
  			   	   producto3 = resta3.mulEscalar(SFi);
  			   	   
  			   	   result = population[actual].sumar(producto);
  			   	   result = result.sumar(producto2);
  			   	   mutant = result.sumar(producto3);
    		       break;
    		       
  		   	  /*// Para hacer esta estrat�gia, lo que hay que elegir es CrossoverType = Arithmetic
  		   	   * case 7: //DE current to rand 1
  		   		   resta = r1.restar(population[actual]); 
  		   		   resta2 = r2.restar(r3);
  		   		 		   		 
  			   	   producto = resta.mulEscalar(RandomGenerator.Randdouble(0, 1));
  			   	   producto2 = resta2.mulEscalar(this.ScalingFactor);
  			   		
  			   	   result = population[actual].sumar(producto);
  			   	   mutant = result.sumar(producto2);
  			   	   
    		       break;
    		       */
  		   }   
  	   

  	  // System.out.println("********Mutante**********");
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
  public double lsff(double Fi, double CRi, PrototypeSet population[][], int [] bestIndividual, int claseObjetivo, int actual, int mejor){
	  PrototypeSet resta, producto, mutant;
	  PrototypeSet crossover;
	  double FitnessFi = 0;
	  
	  
	  //Mutation:
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
   	  mutant = mutant(population[claseObjetivo], actual, mejor, Fi);
   	
   	  
   	  //Crossover
   	  crossover =new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.set(j, mutant.get(j)); // Overwrite.
		   }
	   }
	   
	   
	   // Compute fitness
	   PrototypeSet nominalPopulation = new PrototypeSet();
       nominalPopulation.formatear(crossover);
      
       
       PrototypeSet guardaPopulation  = new PrototypeSet(population[claseObjetivo][actual]);
       
       population[(int) claseObjetivo][actual] = new PrototypeSet(nominalPopulation);
       
       FitnessFi = fitnessFunction(population, bestIndividual ,claseObjetivo,actual); 
       
       
       population[(int) claseObjetivo][actual] = new PrototypeSet(guardaPopulation.clone()); //restarurar
       
       //FitnessFi = accuracy(nominalPopulation,trainingDataSet);
	   
   	   return FitnessFi;
  }
  
  
  
  /**
   * SFGSS local Search.
   * @param population
   * @return
   */
  

  
  public PrototypeSet SFGSS(PrototypeSet population[][], int [] bestIndividual, int claseObjetivo, int actual, int mejor, double CRi){
	  double a=0.1, b=1;
	  double fi1=0, fi2=0, fitnessFi1=0, fitnessFi2=0;
	  double phi = (1+ Math.sqrt(5))/5;
	  double scaling;
	  PrototypeSet crossover, resta, producto, mutant;
	  
	  for (int i=0; i<this.iterSFGSS; i++){ // Computation budjet
	  
		  fi1 = b - (b-a)/phi;
		  fi2 = a + (b-a)/phi;
		  
		  fitnessFi1 = lsff(fi1, CRi, population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi2 = lsff(fi2, CRi,population,bestIndividual, claseObjetivo, actual,mejor);
		  
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
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
	  mutant = mutant(population[claseObjetivo], actual, mejor, scaling);
   	  
   	  //Crossover
   	  crossover =new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.set(j, mutant.get(j)); // Overwrite.
		   }
	   }
	   
	   
	  
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
  
  public  PrototypeSet SFHC(PrototypeSet population[][], int [] bestIndividual, int claseObjetivo,  int actual, int mejor, double SFi, double CRi){
	  double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
	  PrototypeSet crossover, resta, producto, mutant;
	  double h= 0.5;
	  
	  
	  for (int i=0; i<this.iterSFHC; i++){ // Computation budjet
		  		  
		  fitnessFi1 = lsff(SFi-h, CRi, population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi2 = lsff(SFi, CRi,  population,bestIndividual, claseObjetivo,actual,mejor);
		  fitnessFi3 = lsff(SFi+h, CRi,  population,bestIndividual, claseObjetivo,actual,mejor);
		  
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
	  mutant = new PrototypeSet(population[claseObjetivo][actual].size());
	  mutant = mutant(population[claseObjetivo], actual, mejor, SFi);
	 
   	  //Crossover
   	  crossover = new PrototypeSet(population[claseObjetivo][actual]);
   	  
	   for(int j=0; j< population[claseObjetivo][actual].size(); j++){ // For each part of the solution
		   
		   double randNumber = RandomGenerator.Randdouble(0, 1);
			   
		   if(randNumber< CRi){
			   crossover.set(j, mutant.get(j)); // Overwrite.
		   }
	   }
	   
	   
	  
	return crossover;
  
  }

  
  
  
  /**
   * Implements the 1NN algorithm
   * @param current Prototype which the algorithm will find its nearest-neighbor.
   * @param dataSet Prototype set in which the algorithm will search.
   * @return Nearest prototype to current in the prototype set dataset.
   */
  public static Prototype _1nn(Prototype current, PrototypeSet dataSet)
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
           currDist = Distance.euclideanDistance(pi,current);

          
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
  
  public double classficationAccuracy1NN(PrototypeSet training, PrototypeSet test)
  {
	int wellClassificated = 0;
      for(Prototype p : test)
      {
          Prototype nearestNeighbor = _1nn(p, training);          
          
          if(p.getOutput(0) == nearestNeighbor.getOutput(0))
              ++wellClassificated;
      }
  
      
      return 100.0* (wellClassificated / (double)test.size());
  }
  
  
  
  public int[] classify(PrototypeSet training, PrototypeSet test)
  {
	int predicho[] = new int[test.size()];
	
	int i=0;
      for(Prototype p : test)
      {
          Prototype nearestNeighbor = _1nn(p, training);          
          
          predicho[i] = (int) nearestNeighbor.getOutput(0);
                   
          i++;
      }
  
      
      return predicho;
  }
  
  
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
  
  
  
/** Main method */
  
  public PrototypeSet reduceSet(PrototypeSet initial){

	  
 // decide the structure of the particle.
	  	
	  PrototypeSet randomize=new PrototypeSet(initial.clone()) ;
	  generador = new PrototypeGenerator(trainingDataSet);
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
	  
	  
	  // Initial Fitness:
	  PrototypeSet nominalPopulation = new PrototypeSet();
      nominalPopulation.formatear(randomize);
      
      if(randomize.size()>=2){
       System.err.println("\nInitial % de acierto en training Nominal " + classficationAccuracy1NN(randomize,trainingDataSet) );
      }
	  
      
	  // A population per class.
	  
	  PrototypeSet population[][] = new PrototypeSet[this.numberOfClass][PopulationSize];
	  double localAcc[][] = new double[this.numberOfClass][PopulationSize];
	  double Acc[] = new double[PopulationSize];
	  
	  int bestIndividual[] = new int[this.numberOfClass];
	  double bestFitness[] = new double[this.numberOfClass];
	  
	  if(randomize.size() <2){ // If SSMA fails.
		  this.numberOfPrototypes = (int)Math.round(trainingDataSet.size()*0.05);

		  randomize= new PrototypeSet(generador.selecRandomSet(numberOfPrototypes,true).clone()) ;
		  
		  this.numberOfPrototypes = (int)Math.round(trainingDataSet.size()*0.05);
		  
		  // Aseguro que al menos hay un representante de cada clase.
		  PrototypeSet clases[] = new PrototypeSet [this.numberOfClass];
		  for(int i=0; i< this.numberOfClass; i++){
			  clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i));
		  }
		
		  for(int i=0; i< randomize.size(); i++){
			  for(int j=0; j< this.numberOfClass; j++){
				  if(randomize.getFromClass(j).size() ==0 && clases[j].size()!=0){
					  
					  randomize.add(clases[j].getRandom());
				  }
			  }
		  }
	  }
	  
	 
	  // System.out.println("Number of classes " + this.numberOfClass);
	  

	 // randomize.print();
	  
	  // Y ahora divido en los por clases!
	  
	  for(int i=0; i< this.numberOfClass; i++){
		  
		  if(randomize.getFromClass(i).size()>0){
			  population[i][0]= new PrototypeSet(randomize.getFromClass(i).clone());
		   }else{
			   population[i][0] = null; // Sometime it can occur
			   
		   }
	  }
	  
	  
	  // El resto de particulas son iguales!!
	  
		  for(int j=1; j<this.PopulationSize; j++){
	
		  
			  for(int i=0; i< this.numberOfClass; i++){
				  
				  if(population[i][0]!=null){
					  population[i][j]= new PrototypeSet();
					  
					  PrototypeSet deLaClase = new PrototypeSet(trainingDataSet.getFromClass(i).clone());
					  
					  for(int z=0; z<deLaClase.size(); z++){
						  deLaClase.get(z).setIndex(z); // Establish index.
					  }
					
					  ArrayList<Integer> lista = RandomGenerator.generateDifferentRandomIntegers(0, deLaClase.size()-1);
					  
					  for(int z=0; z< population[i][0].size(); z++){
						  population[i][j].add(deLaClase.get(lista.get(z)));
					  }
		  		  }	  
			  }
		  }
		  

	  // Calculate initial fitness
	  
	  for(int i=0; i< this.numberOfClass; i++){
		  bestIndividual[i] = 0;
		  bestFitness[i] = Double.MIN_VALUE;
		  
		  if(population[i][0]!=null){
			  for(int j=0; j<this.PopulationSize; j++){
				  
				  double fitness= initialfitnessFunction(population, i, j);
				  
				  
				  if(fitness > bestFitness[i]){
					  bestFitness[i] = fitness;
					  bestIndividual[i] = j;
				  }
				  
				  
			  }
		  }
	  }
	  
	  
    
	  // Co-evolutionary stage
	  
	  int iter =0;
	  
		while(iter<MAX_ITER){
			
			for(int i=0; i<this.numberOfClass;i++){
				 // Do generation...population[i][j]
				if(population[i][0]!=null){
					population[i]= doGeneration(population, i, bestIndividual);	
				}
			}
			
			//updateCollaborators; // who is the best ?
			
			  for(int i=0; i< this.numberOfClass; i++){
				  
				  if(population[i][0]!=null){
					  
				 
					  for(int j=0; j<this.PopulationSize; j++){  //REPASAR
						  
						  double fitness= fitnessFunction(population, bestIndividual, i, j);
						  
						  if(fitness > bestFitness[i]){
							  bestFitness[i] = fitness;
							  bestIndividual[i] = j;
						  }
						  
						  
					  }
				  
				  }
			  }
			  
			  
			
			iter++;
		}
	  
		
		// Generate Final reference set.
		
		
		  PrototypeSet Join = new PrototypeSet();
		  for(int i=0; i<this.numberOfClass;i++){
			  if(population[i][0]!=null){
				  Join.add(population[i][bestIndividual[i]]); 
			  }
		  }
		

		   nominalPopulation = new PrototypeSet();
           nominalPopulation.formatear(Join);
           
    
         System.err.println("\n% de acierto en training Nominal " + classficationAccuracy1NN(nominalPopulation,trainingDataSet) );
    	 System.out.println("Reduction % " + (100.-(nominalPopulation.size()*100.)/trainingDataSet.size()) );
           
		
    	 
		  int pre[]= classify(nominalPopulation, trainingDataSet);
		  double tprate= this.TPrate(trainingDataSet, pre);
		  double fprate= this.FPrate(trainingDataSet, pre);
		  
		  System.out.println("AUC = "+(1.0+tprate-fprate)/2.0);;
		  
		  
	  
	  return nominalPopulation;
  }

  

  /**
   * Initial fitness fuction
   * @param population
   * @param claseObjetivo
   * @param particle
   * @return
   */
  
  public double initialfitnessFunction(PrototypeSet population[][], double claseObjetivo, int particle){
	  double fitness =0;
	  
	  PrototypeSet Join = new PrototypeSet();
	  for(int i=0; i<this.numberOfClass;i++){
		  if(population[i][0]!=null){
			  Join.add(population[i][particle]);
		  }
	  }

	  double acc[] = new double[this.numberOfClass];
	  double global = 1;
	  
	  
	  for(int i=0; i<this.numberOfClass;i++){
		  acc[i] = classficationAccuracy1NN(Join,trainingDataSet.getFromClass(i)); // local accss
	  }
	  
	  if(this.tipoFitness.equalsIgnoreCase("Weighted")){
		  global = classficationAccuracy1NN(Join,trainingDataSet);
		  
	  }else if(this.tipoFitness.equalsIgnoreCase("GeometricMean")){
		 // AccLocal: Acc1
		  // AccGlobal:  Math.sqrt(Acc0*Acc1*Acc2)
		  for(int i=0; i<this.numberOfClass;i++){
			  global *= acc[i];
		  }
		  global = Math.sqrt(global);
		  
	  }else if(this.tipoFitness.equalsIgnoreCase("AUC")){
		  // (1+TPrate+FPrate)/2
		  int pre[]= classify(Join, trainingDataSet);
		  double tprate= this.TPrate(trainingDataSet, pre);
		  double fprate= this.FPrate(trainingDataSet, pre);
		  
		  global = (1.0+tprate-fprate)/2.0;
	  }
	 
	  fitness =  global;//;this.Beta*global+ (1-this.Beta)*acc[(int) claseObjetivo]; 
	  
	  return fitness;
	  
  }
  
  /**
   * Fitness function
   * @param population
   * @param bestIndividual
   * @param claseObjetivo
   * @param particle
   * @return
   */
  
  public double fitnessFunction(PrototypeSet population[][], int bestIndividual[], double claseObjetivo, int particle){
	  double fitness =0;
	  
	  if(population[(int) claseObjetivo][0]!=null){
	  
		  PrototypeSet Join = new PrototypeSet(population[(int) claseObjetivo][particle]);
		  
		  for(int i=0; i<this.numberOfClass;i++){
			  if(i!= claseObjetivo && population[i][0]!=null ){
				  Join.add(population[i][bestIndividual[i]]);  
			  }
		  }
		  
		  double acc[] = new double[this.numberOfClass];
		  double global = 1;
		  
		  

		  if(this.tipoFitness.equalsIgnoreCase("Weighted")){
			  global = classficationAccuracy1NN(Join,trainingDataSet);
		  }else if(this.tipoFitness.equalsIgnoreCase("GeometricMean")){
			  for(int i=0; i<this.numberOfClass;i++){
				  acc[i] = classficationAccuracy1NN(Join,trainingDataSet.getFromClass(i)); // local accss
			  }
			  
			 // AccLocal: Acc1
			  // AccGlobal:  Math.sqrt(Acc0*Acc1*Acc2)
			  for(int i=0; i<this.numberOfClass;i++){
				  global *= acc[i];
			  }
			  global = Math.sqrt(global);
			  
		  }else if(this.tipoFitness.equalsIgnoreCase("AUC")){
			  // (1+TPrate+FPrate)/2
			  int pre[]= classify(Join, trainingDataSet);
			  double tprate= this.TPrate(trainingDataSet, pre);
			  double fprate= this.FPrate(trainingDataSet, pre);
			  
			  global = (1.0+tprate-fprate)/2.0;
		  }
		 
		  fitness = global; //this.Beta*global+ (1-this.Beta)*acc[(int) claseObjetivo]; 
		  
		  
		  
	  }
	  return fitness;
	  
  }
  
  /**
   * Generate a reduced prototype set by the CoDEGenerator method.
   * @return Reduced set by CoDEGenerator's method.
   */
  
  
  public PrototypeSet[] doGeneration(PrototypeSet population[][], double claseObjetivo, int bestIndividual[])
  {
	  //Algorithm

	  PrototypeSet nominalPopulation;

	  PrototypeSet mutation[] = new PrototypeSet[PopulationSize];
	  PrototypeSet crossover[] = new PrototypeSet[PopulationSize];
	    
	  double ScalingFactor[] = new double[this.PopulationSize];
	  double CrossOverRate[] = new double[this.PopulationSize]; // Inside of the Optimization process.
	  double fitness[] = new double[PopulationSize];
	  
	  
	  // Calculate fitness function for each particle
  
	  for(int i=0; i< PopulationSize; i++){
		  fitness[i] =fitnessFunction(population, bestIndividual ,claseObjetivo,i); 
	  }
	  
	  
	  //We select the best initial  particle
	 double bestFitness=fitness[0];
	  int bestFitnessIndex=0;
	  for(int i=1; i< PopulationSize;i++){
		  if(fitness[i]>bestFitness){
			  bestFitness = fitness[i];
			  bestFitnessIndex=i;
		  }
		  
	  }
	  
	   for(int j=0;j<PopulationSize;j++){
         //Now, I establish the index of each prototype.
		   if(population[(int) claseObjetivo][0]!=null){
		   
			   for(int i=0; i<population[(int) claseObjetivo][j].size(); ++i)
				   population[(int) claseObjetivo][j].get(i).setIndex(i);
		   }
	   }
      
	   boolean cruceExp [] = new boolean[PopulationSize];
	   
	   
	   // Initially the Scaling Factor and crossover for each Individual are randomly generated between 0 and 1.
	   
	   for(int i=0; i< this.PopulationSize; i++){
		   ScalingFactor[i] =  RandomGenerator.Randdouble(0, 1);
		   CrossOverRate[i] =  RandomGenerator.Randdouble(0, 1);
	   }
	   
	   
	  	   
	   double randj[] = new double[5];
	   
	   for(int iter=0; iter< MaxIter; iter++){ // Main loop
		      
		   for(int i=0; i<PopulationSize; i++){

			   // Generate randj for j=1 to 5.
			   for(int j=0; j<5; j++){
				   randj[j] = RandomGenerator.Randdouble(0, 1);
			   }
			   
					   
    			   	    
			   
			   if(i==bestFitnessIndex && randj[4] < tau[2]){
				  // System.out.println("SFGSS applied");
				   //SFGSS
				   crossover[i] = SFGSS(population, bestIndividual, (int) claseObjetivo, i, bestFitnessIndex, CrossOverRate[i]);
				   
				   
			   }else if(i==bestFitnessIndex &&  tau[2] <= randj[4] && randj[4] < tau[3]){
				   //SFHC
				   //System.out.println("SFHC applied");
				   crossover[i] = SFHC(population, bestIndividual, (int) claseObjetivo, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i]);
				   
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
				   
				   mutation[i] = new PrototypeSet(population[(int) claseObjetivo][i].size());
			   
				  //Mutation:
					
				   mutation[i]  = mutant(population[(int) claseObjetivo], i, bestFitnessIndex, ScalingFactor[i]);
				   
				    // Crossver Operation.

				   crossover[i] = new PrototypeSet(population[(int) claseObjetivo][i]);
				   
				   for(int j=0; j< population[(int) claseObjetivo][i].size(); j++){ // For each part of the solution
					   
					   double randNumber = RandomGenerator.Randdouble(0, 1);
						   
					   if(randNumber<CrossOverRate[i]){
						   crossover[i].set(j, mutation[i].get(j)); // Overwrite.
					   }
				   }
				   
				   
				   
				   
			   }
			   
   
			   
			   // Fourth: Selection Operation.
		   
			   nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(population[(int) claseObjetivo][i]);
		      // fitness[i] = accuracy(nominalPopulation,trainingDataSet.getFromClass(claseObjetivo));
		       
		       PrototypeSet guardaPopulation  = new PrototypeSet(population[(int) claseObjetivo][i]);
		      
		       population[(int) claseObjetivo][i] = new PrototypeSet(nominalPopulation);
		       
		       fitness[i] = fitnessFunction(population, bestIndividual ,claseObjetivo,i); 
		       
		       nominalPopulation = new PrototypeSet();
		       nominalPopulation.formatear(crossover[i]);
		       
		       population[(int) claseObjetivo][i] = new PrototypeSet(nominalPopulation);
		       
			   double trialVector = fitnessFunction(population, bestIndividual ,claseObjetivo,i); //accuracy(nominalPopulation,trainingDataSet.getFromClass(claseObjetivo));
			
			   
			   
		  
			  if(trialVector > fitness[i]){
				  population[(int) claseObjetivo][i] = new PrototypeSet(crossover[i]);
				  fitness[i] = trialVector;
			  }else{
				  population[(int) claseObjetivo][i] = new PrototypeSet(guardaPopulation); // restitutyo
			  }
			  
			  /*
			    if(fitness[i]>bestFitness){
				  bestFitness = fitness[i];
				  bestFitnessIndex=i;
				  //System.out.println("Iter="+ iter +" Acc= "+ bestFitness);
			  }
			  */
			  
			  
		   }

		   //System.out.println("Acc= "+ bestFitness);
	   }

	   
		 //  nominalPopulation = new PrototypeSet();
          // nominalPopulation.formatear(population[bestFitnessIndex]);
		//  System.err.println("\n% de acierto en training Nominal " + KNN.classficationAccuracy(nominalPopulation,trainingDataSet,1)*100./trainingDataSet.size() );
			  
			//  nominalPopulation.print();

  
		return population[(int) claseObjetivo];
  }
  
  
  
  
  
  /* MEzcla de algoritmos */
  public void ejecutar () {

    int i, j, l;
    double conjS[][];
    double conjR[][];
    int conjN[][];
    boolean conjM[][];
    int clasesS[];
    int nSel = 0;
    Cromosoma poblacion[];
    double ev = 0;
    double dMatrix[][];
    int sel1, sel2, comp1, comp2;
    Cromosoma hijos[];
    double umbralOpt;
    boolean veryLarge;
    double GAeffort=0, LSeffort=0, temporal;
    double fAcierto=0, fReduccion=0;
    int contAcierto=0, contReduccion=0;
    int nClases;

    long tiempo = System.currentTimeMillis();

    /*Getting the number of different classes*/
    nClases = 0;
    for (i=0; i<clasesTrain.length; i++)
      if (clasesTrain[i] > nClases)
        nClases = clasesTrain[i];
    nClases++;

    if (datosTrain.length > 9000) {
      veryLarge = true;
    } else {
      veryLarge = false;
    }

    if (veryLarge == false) {
      /*Construct a distance matrix of the instances*/
      dMatrix = new double[datosTrain.length][datosTrain.length];
      for (i = 0; i < dMatrix.length; i++) {
        for (j = i + 1; j < dMatrix[i].length; j++) {
          dMatrix[i][j] = KNN.distancia(datosTrain[i], realTrain[i], nominalTrain[i], nulosTrain[i], datosTrain[j], realTrain[j], nominalTrain[j], nulosTrain[j], distanceEu);
        }
      }
      for (i = 0; i < dMatrix.length; i++) {
        dMatrix[i][i] = Double.POSITIVE_INFINITY;
      }
      for (i = 0; i < dMatrix.length; i++) {
        for (j = i - 1; j >= 0; j--) {
          dMatrix[i][j] = dMatrix[j][i];
        }
      }
    } else {
      dMatrix = null;
    }

    /*Random inicialization of the population*/
    Randomize.setSeed (semilla);
    poblacion = new Cromosoma[tamPoblacion];
    for (i=0; i<tamPoblacion; i++)
      poblacion[i] = new Cromosoma (kNeigh, datosTrain.length, dMatrix, datosTrain, realTrain, nominalTrain, nulosTrain, distanceEu);

    /*Initial evaluation of the population*/
    for (i=0; i<tamPoblacion; i++) {
      poblacion[i].evaluacionCompleta(nClases, kNeigh, clasesTrain);
    }

    umbralOpt = 0;

    /*Until stop condition*/
    while (ev < nEval) {

      Arrays.sort(poblacion);

      if (fAcierto >= (double)poblacion[0].getFitnessAc()*100.0/(double)datosTrain.length) {
        contAcierto++;
      } else {
        contAcierto=0;
      }
      fAcierto = (double)poblacion[0].getFitnessAc()*100.0/(double)datosTrain.length;

      if (fReduccion >= (1.0-((double)poblacion[0].genesActivos()/(double)datosTrain.length))*100.0) {
        contReduccion++;
      } else {
        contReduccion=0;
      }
      fReduccion = (1.0-((double)poblacion[0].genesActivos()/(double)datosTrain.length))*100.0;

      if (contReduccion >= 10 || contAcierto >= 10){
        if (Randomize.Randint(0,1)==0) {
          if (contAcierto >= 10) {
            contAcierto = 0;
            umbralOpt++;
          } else {
            contReduccion = 0;
            umbralOpt--;
          }
        } else {
          if (contReduccion >= 10) {
            contReduccion = 0;
            umbralOpt--;
          } else {
            contAcierto = 0;
            umbralOpt++;
          }
        }
      }

      /*Binary tournament selection*/
      comp1 = Randomize.Randint(0,tamPoblacion-1);
      do {
        comp2 = Randomize.Randint(0,tamPoblacion-1);
      } while (comp2 == comp1);

      if (poblacion[comp1].getFitness() > poblacion[comp2].getFitness())
        sel1 = comp1;
      else sel1 = comp2;
      comp1 = Randomize.Randint(0,tamPoblacion-1);
      do {
        comp2 = Randomize.Randint(0,tamPoblacion-1);
      } while (comp2 == comp1);
      if (poblacion[comp1].getFitness() > poblacion[comp2].getFitness())
        sel2 = comp1;
      else
        sel2 = comp2;


      hijos = new Cromosoma[2];
      hijos[0] = new Cromosoma (kNeigh, poblacion[sel1], poblacion[sel2], pCross,datosTrain.length);
      hijos[1] = new Cromosoma (kNeigh, poblacion[sel2], poblacion[sel1], pCross,datosTrain.length);
      hijos[0].mutation (kNeigh, pMut, dMatrix, datosTrain, realTrain, nominalTrain, nulosTrain, distanceEu);
      hijos[1].mutation (kNeigh, pMut, dMatrix, datosTrain, realTrain, nominalTrain, nulosTrain, distanceEu);

      /*Evaluation of offsprings*/
      hijos[0].evaluacionCompleta(nClases, kNeigh, clasesTrain);
      hijos[1].evaluacionCompleta(nClases, kNeigh, clasesTrain);
      ev+=2;
      GAeffort += 2;
      temporal = ev;
      if (hijos[0].getFitness() > poblacion[tamPoblacion-1].getFitness() || Randomize.Rand() < 0.0625) {
    	  ev += hijos[0].optimizacionLocal(nClases, kNeigh, clasesTrain,dMatrix,umbralOpt, datosTrain, realTrain, nominalTrain, nulosTrain, distanceEu);
      }
      if (hijos[1].getFitness() > poblacion[tamPoblacion-1].getFitness() || Randomize.Rand() < 0.0625) {
          ev += hijos[1].optimizacionLocal(nClases, kNeigh, clasesTrain,dMatrix,umbralOpt, datosTrain, realTrain, nominalTrain, nulosTrain, distanceEu);
      }

      LSeffort += (ev - temporal);

      /*Replace the two worst*/
      if (hijos[0].getFitness() > poblacion[tamPoblacion-1].getFitness()) {
        poblacion[tamPoblacion-1] = new Cromosoma (kNeigh, datosTrain.length, hijos[0]);
      }
      if (hijos[1].getFitness() > poblacion[tamPoblacion-2].getFitness()) {
        poblacion[tamPoblacion-2] = new Cromosoma (kNeigh, datosTrain.length, hijos[1]);
      }

/*      System.out.println(ev + " - (" + umbralOpt + ")" +
                         (double)poblacion[0].getFitnessAc()*100.0/(double)datosTrain.length + " / " +
                         (double)poblacion[tamPoblacion-1].getFitnessAc()*100.0/(double)datosTrain.length + " - " +
                         (1.0-((double)poblacion[0].genesActivos()/(double)datosTrain.length))*100.0 + " / " +
                         (1.0-((double)poblacion[tamPoblacion-1].genesActivos()/(double)datosTrain.length))*100.0);*/
    }

    Arrays.sort(poblacion);
    nSel = poblacion[0].genesActivos();

    /*Construction of S set from the best cromosome*/
    conjS = new double[nSel][datosTrain[0].length];
    conjR = new double[nSel][datosTrain[0].length];
    conjN = new int[nSel][datosTrain[0].length];
    conjM = new boolean[nSel][datosTrain[0].length];
    clasesS = new int[nSel];
    for (i=0, l=0; i<datosTrain.length; i++) {
      if (poblacion[0].getGen(i)) { //the instance must be copied to the solution
        for (j=0; j<datosTrain[i].length; j++) {
          conjS[l][j] = datosTrain[i][j];
          conjR[l][j] = realTrain[i][j];
          conjN[l][j] = nominalTrain[i][j];
          conjM[l][j] = nulosTrain[i][j];
        }
        clasesS[l] = clasesTrain[i];
        l++;
      }
    }

    System.out.println("SSMA "+ relation + " " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS, entradas, salida, nEntradas, relation);
    OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, nEntradas, relation);
    
    
    /** AHORA A�ADO MI DE!! **/
    Parameters.assertBasicArgs(ficheroSalida);
    
    PrototypeGenerationAlgorithm.readParametersFile(this.Script);
    PrototypeGenerationAlgorithm.printParameters();
    
    PrototypeSet training = readPrototypeSet(ficheroSalida[0]);
    
    
 //   training.print(); // Conjunto devuelto POR SSMA
    
    trainingDataSet = readPrototypeSet(this.ficheroTraining); // Conjunto inicial
    testDataSet = readPrototypeSet(this.ficheroTest);
    
   // trainingDataSet.print();
     //this.numberOfPrototypes = (int)Math.floor((trainingDataSet.size())*ParticleSize/100.0);
 
     PrototypeSet DE = reduceSet(training); // LLAMO al SADE
	    DE.save(ficheroSalida[0]); // Lo guardo
        //Copy the test input file to the output test file
       // KeelFile.copy(inputFilesPath.get(TEST), outputFilesPath.get(TEST));
	    System.out.println("Time elapse:" + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");
	    
	    
	    /*ADDING KNN FOR TEST FILE */
        int trainRealClass[][];
        int trainPrediction[][];
            
        trainRealClass = new int[datosTrain.length][1];
    trainPrediction = new int[datosTrain.length][1];	
       
         nClases = DE.getPosibleValuesOfOutput().size();
             
       //Working on training
        for (i=0; i<trainingDataSet.size(); i++) {
             trainRealClass[i][0] = (int) trainingDataSet.get(i).getOutput(0);
             trainPrediction[i][0] = evaluate(trainingDataSet.get(i).getInputs(),DE.prototypeSetTodouble(), nClases, DE.getClases(), 1);
        }
        
        Attribute entradas[];
    Attribute salida;
                     
        entradas = Attributes.getInputAttributes();
        salida = Attributes.getOutputAttribute(0);
        String relation =  Attributes.getRelationName(); 
            
        writeOutput(this.ficheroSalida[0], trainRealClass, trainPrediction, entradas, salida, relation);
        
        int realClass[][] = new int[datosTest.length][1];
    int prediction[][] = new int[datosTest.length][1];	
	
			
        for (i=0; i<realClass.length; i++) {
	realClass[i][0] = (int) testDataSet.get(i).getOutput(0);
	prediction[i][0]= evaluate(testDataSet.get(i).getInputs(),DE.prototypeSetTodouble(), nClases, DE.getClases(), 1);
        }
            
        writeOutput(this.ficheroSalida[1], realClass, prediction,  entradas, salida, relation);
        
        
        
  }
  
  /**
	 * Prints output files.
	 * 
	 * @param filename Name of output file
	 * @param realClass Real output of instances
	 * @param prediction Predicted output for instances
	 */
	public static void writeOutput(String filename, int [][] realClass, int [][] prediction, Attribute inputs[], Attribute output, String relation) {
	
		String text = "";
		
                  
		/*Printing input attributes*/
		text += "@relation "+ relation +"\n";

		for (int i=0; i<inputs.length; i++) {
			
			text += "@attribute "+ inputs[i].getName()+" ";
			
		    if (inputs[i].getType() == Attribute.NOMINAL) {
		    	text += "{";
		        for (int j=0; j<inputs[i].getNominalValuesList().size(); j++) {
		        	text += (String)inputs[i].getNominalValuesList().elementAt(j);
		        	if (j < inputs[i].getNominalValuesList().size() -1) {
		        		text += ", ";
		        	}
		        }
		        text += "}\n";
		    } else {
		    	if (inputs[i].getType() == Attribute.INTEGER) {
		    		text += "integer";
		        } else {
		        	text += "real";
		        }
		        text += " ["+String.valueOf(inputs[i].getMinAttribute()) + ", " +  String.valueOf(inputs[i].getMaxAttribute())+"]\n";
		    }
		}

		/*Printing output attribute*/
		text += "@attribute "+ output.getName()+" ";

		if (output.getType() == Attribute.NOMINAL) {
			text += "{";
			
			for (int j=0; j<output.getNominalValuesList().size(); j++) {
				text += (String)output.getNominalValuesList().elementAt(j);
		        if (j < output.getNominalValuesList().size() -1) {
		        	text += ", ";
		        }
			}		
			text += "}\n";	    
		} else {
		    text += "integer ["+String.valueOf(output.getMinAttribute()) + ", " + String.valueOf(output.getMaxAttribute())+"]\n";
		}

		/*Printing data*/
		text += "@data\n";

		Files.writeFile(filename, text);
		
		if (output.getType() == Attribute.INTEGER) {
			
			text = "";
			
			for (int i=0; i<realClass.length; i++) {
			      
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + realClass[i][j] + " ";
			      }
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + prediction[i][j] + " ";
			      }
			      text += "\n";			      
			      if((i%10)==9){
			    	  Files.addToFile(filename, text);
			    	  text = "";
			      }     
			}			
			
			if((realClass.length%10)!=0){
				Files.addToFile(filename, text);
			}
		}
		else{
			
			text = "";
			
			for (int i=0; i<realClass.length; i++) {
			      
			      for (int j=0; j<realClass[0].length; j++){
			    	  text += "" + (String)output.getNominalValuesList().elementAt(realClass[i][j]) + " ";
			      }
			      for (int j=0; j<realClass[0].length; j++){
			    	  if(prediction[i][j]>-1){
			    		  text += "" + (String)output.getNominalValuesList().elementAt(prediction[i][j]) + " ";
			    	  }
			    	  else{
			    		  text += "" + "Unclassified" + " ";
			    	  }
			      }
			      text += "\n";
			      
			      if((i%10)==9){
			    	  Files.addToFile(filename, text);
			    	  text = "";
			      } 
			}			
			
			if((realClass.length%10)!=0){
				Files.addToFile(filename, text);
			}		
		}
		
	}//end-method
      
  
  /** 
	 * Calculates the Euclidean distance between two instances
	 * 
	 * @param instance1 First instance 
	 * @param instance2 Second instance
	 * @return The Euclidean distance
	 * 
	 */
	protected static double distance(double instance1[],double instance2[]){
		
		double length=0.0;

		for (int i=0; i<instance1.length; i++) {
			length += (instance1[i]-instance2[i])*(instance1[i]-instance2[i]);
		}
			
		length = Math.sqrt(length); 
				
		return length;
		
	} //end-method
      
           
  /** 
	 * Evaluates a instance to predict its class.
	 * 
	 * @param example Instance evaluated 
	 * @return Class predicted
	 * 
	 */
	public static int evaluate (double example[], double trainData[][],int nClasses,int trainOutput[],int k) {
	
		double minDist[];
		int nearestN[];
		int selectedClasses[];
		double dist;
		int prediction;
		int predictionValue;
		boolean stop;

		nearestN = new int[k];
		minDist = new double[k];
	
	    for (int i=0; i<k; i++) {
			nearestN[i] = 0;
			minDist[i] = Double.MAX_VALUE;
		}
		
	    //KNN Method starts here
	    
		for (int i=0; i<trainData.length; i++) {
		
		    dist = distance(trainData[i],example);

			if (dist > 0.0){ //leave-one-out
			
				//see if it's nearer than our previous selected neighbors
				stop=false;
				
				for(int j=0;j<k && !stop;j++){
				
					if (dist < minDist[j]) {
					    
						for (int l = k - 1; l >= j+1; l--) {
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
		
		//we have check all the instances... see what is the most present class
		selectedClasses= new int[nClasses];
	
		for (int i=0; i<nClasses; i++) {
			selectedClasses[i] = 0;
		}	
		
		for (int i=0; i<k; i++) {
               //      System.out.println("nearestN i ="+i + " =>"+nearestN[i]);
                // System.out.println("trainOutput ="+trainOutput[nearestN[i]]);
                  
			selectedClasses[trainOutput[nearestN[i]]]+=1;
		}
		
		prediction=0;
		predictionValue=selectedClasses[0];
		
		for (int i=1; i<nClasses; i++) {
		    if (predictionValue < selectedClasses[i]) {
		        predictionValue = selectedClasses[i];
		        prediction = i;
		    }
		}
		
		return prediction;
	
	} //end-method	
	

  public void leerConfiguracion (String ficheroScript) {

    String fichero, linea, token;
    StringTokenizer lineasFichero, tokens;
    byte line[];
    int i, j;

    ficheroSalida = new String[2];

    fichero = Fichero.leeFichero (ficheroScript);
    lineasFichero = new StringTokenizer (fichero,"\n\r");

    lineasFichero.nextToken();
    linea = lineasFichero.nextToken();

    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the name of training and test files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTraining = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTest = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroTest = new String (line,i,j-i);


    //Parameters.assertBasicArgs(ficheroSalida);
    
    

    
    /*Obtainin the path and the base name of the results files*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    token = tokens.nextToken();

    /*Getting the name of output files*/
    line = token.getBytes();
    for (i=0; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[0] = new String (line,i,j-i);
    for (i=j+1; line[i]!='\"'; i++);
    i++;
    for (j=i; line[j]!='\"'; j++);
    ficheroSalida[1] = new String (line,i,j-i);

    /*Getting the seed*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    semilla = Long.parseLong(tokens.nextToken().substring(1));

    /*Getting the size of the poblation and the number of evaluations*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    tamPoblacion = Integer.parseInt(tokens.nextToken().substring(1));
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    nEval = Double.parseDouble(tokens.nextToken().substring(1));

    /*Getting the probabilities of evolutionary operators*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    pCross = Double.parseDouble(tokens.nextToken().substring(1));
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    pMut = Double.parseDouble(tokens.nextToken().substring(1));

    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    kNeigh = Integer.parseInt(tokens.nextToken().substring(1));
 
    /*Getting the type of distance function*/
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;    
    
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.MAX_ITER = Integer.parseInt(tokens.nextToken().substring(1));
    
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.PopulationSize = Integer.parseInt(tokens.nextToken().substring(1));
    
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.MaxIter = Integer.parseInt(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.iterSFGSS = Integer.parseInt(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.iterSFHC = Integer.parseInt(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.Fl =  Double.parseDouble(tokens.nextToken().substring(1));
    
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.Fu =  Double.parseDouble(tokens.nextToken().substring(1));
    
    tau = new double[4];
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.tau[0] =  Double.parseDouble(tokens.nextToken().substring(1));
   
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.tau[1] =  Double.parseDouble(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.tau[2] =  Double.parseDouble(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.tau[3] =  Double.parseDouble(tokens.nextToken().substring(1));
    
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.Strategy = Integer.parseInt(tokens.nextToken().substring(1));
    
  
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.Beta =  Double.parseDouble(tokens.nextToken().substring(1));
    
    linea = lineasFichero.nextToken();
    tokens = new StringTokenizer (linea, "=");
    tokens.nextToken();
    this.tipoFitness = tokens.nextToken().substring(1);     
    
    
    System.out.print("\nIsaac dice: tipo fitnes ="+this.tipoFitness +"\n");

    
    
    
  }
}
