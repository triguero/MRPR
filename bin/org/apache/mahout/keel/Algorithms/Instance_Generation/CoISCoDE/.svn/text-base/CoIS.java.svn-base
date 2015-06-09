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
import java.util.StringTokenizer;

import org.core.*;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.OutputIS;

import org.apache.mahout.keel.Algorithms.Coevolution.CoevolutionAlgorithm;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerator;

import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;

/**
Numero de poblaciones: 1 por clase
Tamaño de la poblacion: 30 (50 es mucho)
percentage0to1HUX: 0.25    (el numero de 1's que se generan con el operador HUX)
Numero de 1's en los cromosomas al inicializar: 25%
Parametros del fitness: alpha (peso entre reduccion y accuracy): 0.5   Beta (peso entre Acc local y global): 0.5
**/

public class CoIS extends CoevolutionAlgorithm{

	private int K;

	private int MAX_EVALUATIONS;
	private int evaluations;

	private int trainRealClass[][];
	private int trainPrediction[][];
	private int testRealClass[][];
	private int testPrediction[][];	
	private int testUnclassified;	
	private int trainUnclassified;	
	private int testConfMatrix[][];
	private int trainConfMatrix[][];
	
	private Population populations[];
	private static Chromosome collaborators[];
	
	private static int trainSize;
	private static int evals;
	
	private int classDistribution [];
	
	private String Script;
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
	  
	private int oldNClasses;
	private int arrayIS [];
	
	/** 
	 * The main method of the class
	 * 
	 * @param script Name of the configuration script  
	 * 
	 */
	public CoIS (String script) {
		
		this.Script = new String(script);
		
		readDataFiles(script);
		
		//Naming the algorithm
		name="CoIS";

		evals=0;

		int sizePop=20;
		
		//Initializations stuff 
		Translator.initialize();
		
		for(int i=0;i<trainOutput.length;i++){
			Translator.learnValue(trainOutput[i]);
		}
		
		//translation of output attribute for safety
		for(int i=0;i<trainOutput.length;i++){
			trainOutput[i]=Translator.real2safe(trainOutput[i]);
		}
		
		for(int i=0;i<testOutput.length;i++){
			testOutput[i]=Translator.real2safe(testOutput[i]);
		}
		
		oldNClasses=nClasses;
		nClasses=Translator.getSafeClasses();
		
		K=1;
		
		KNN.setK(K);
		KNN.setClasses(nClasses);
		KNN.setData(trainData,trainOutput);

		Population.setSize(sizePop);
		
		//Population.setInitProb(0.25);
		Population.setInitProb(0.2);
		
		populations=new Population [nClasses];
		collaborators=new Chromosome [nClasses];
		
		classDistribution= new int [nClasses];
		
		Arrays.fill(classDistribution, 0);
		
		for(int i=0;i<trainOutput.length;i++){
			classDistribution[trainOutput[i]]++;
		}
		
		for(int i=0; i<populations.length;i++){
			
			populations[i]=new Population(i,classDistribution[i]);

		}
		
		for(int i=0; i<populations.length;i++){
			
			collaborators[i]=populations[i].getRandom();

		}
		
		trainSize=trainData.length;
		
		//Initialization of random generator
	    
	    Randomize.setSeed(seed);
	    
	    //Initialization stuff ends here. So, we can start time-counting
		
		Timer.resetTime();
		
	} //end-method 
	
	/** 
	 * Reads configuration script, to extract the parameter's values.
	 * 
	 * @param script Name of the configuration script  
	 * 
	 */	
	protected void readParameters (String script) {
		
		String file;
		String line;
		StringTokenizer fileLines, tokens;
		
	    file = Files.readFile (script);
	    fileLines = new StringTokenizer (file,"\n\r");
	    
	    //Discard in/out files definition
	    fileLines.nextToken();
	    fileLines.nextToken();
	    fileLines.nextToken();
	    
	    //Getting the seed
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    seed = Long.parseLong(tokens.nextToken().substring(1));
    
	    //Getting the MAX EVALUATIONS parameter
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    MAX_EVALUATIONS = Integer.parseInt(tokens.nextToken().substring(1));
    
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.MAX_ITER = Integer.parseInt(tokens.nextToken().substring(1));
	    
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.PopulationSize = Integer.parseInt(tokens.nextToken().substring(1));
	    
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.MaxIter = Integer.parseInt(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.iterSFGSS = Integer.parseInt(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.iterSFHC = Integer.parseInt(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.Fl =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.Fu =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    tau = new double[4];
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.tau[0] =  Double.parseDouble(tokens.nextToken().substring(1));
	   
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.tau[1] =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.tau[2] =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.tau[3] =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.Strategy = Integer.parseInt(tokens.nextToken().substring(1));
	    
	  
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.Beta =  Double.parseDouble(tokens.nextToken().substring(1));
	    
	    line = fileLines.nextToken();
	    tokens = new StringTokenizer (line, "=");
	    tokens.nextToken();
	    this.tipoFitness = tokens.nextToken().substring(1);     
	    
	    
	    System.out.print("\nIsaac dice: beta ="+this.Beta +"\n");
	    
	    
	}//end-method

	public void coevolution(){
		
		
		//First evaluation
		for(int i=0; i<populations.length;i++){
			
			populations[i].evaluatePopulation();

		}
		System.out.println(evals);
		while(evals<MAX_EVALUATIONS){
						System.out.println(evals);
			for(int i=0; i<populations.length;i++){
				
				populations[i].doGeneration();

			}
			
			updateCollaborators();
			
		}

		generateFinalReferenceSet();
		
		/*Copy final reference set to a file*/
        int selected=0;
       
        for(int i=0;i<arrayIS.length;i++){
           
            if(arrayIS[i]==1){
                selected++;
            }
        }

        double saveData [][]= new double[selected][trainData[0].length];
        int saveOutput []= new int[selected];
        int pointer=0;
       
        for(int c=0, l=0;c<nClasses;c++){       
            for(int i=0;i<trainData.length;i++){
                if(trainOutput[i]==c){
                    if (arrayIS[pointer]==1) { //the instance will be copied to the solution

                        for (int j=0; j<trainData[0].length; j++) {
                            saveData[l][j] = trainData[i][j];
                        }
                        saveOutput[l] = trainOutput[i];
                        
                        System.out.println(saveOutput[l]);
                        l++;
                    }
                    pointer++;
                }
            }
        }
       
		
        /** AHORA A�ADO MI DE!! **/

	  OutputIS.escribeSalida("salida.dat", saveData, saveOutput, inputs, output, inputAtt, "Isaak");
     /*   PrototypeSet training  = readPrototypeSet("salida.dat");
       */
	    
        PrototypeSet training  = new PrototypeSet(saveData.length); //
        training.doubleToprototypeSet(saveData,saveOutput);
        
        training.get(0).setAttributesTypes(inputs);

       

        trainingDataSet =  new PrototypeSet(); //readPrototypeSet(this.trainFile);
        trainingDataSet.doubleToprototypeSet(trainData,trainOutput);
        
        testDataSet = new PrototypeSet();
        testDataSet.doubleToprototypeSet(testData,testOutput);
        
      
        //generador = new PrototypeGenerator(trainingDataSet);
      //Distance.setNumberOfInputs(training.get(0).getInputs().length);
      
        
        System.out.println("Initial acc = " +classficationAccuracy1NN(training, trainingDataSet));
        
       // training.print(); // Conjunto devuelto POR CoIS
       
 
        
       // trainingDataSet.print();
         //this.numberOfPrototypes = (int)Math.floor((trainingDataSet.size())*ParticleSize/100.0);
     
         	PrototypeSet DE = new PrototypeSet(reduceSet(training.clone())); // LLAMO al SADE

         	DE.print();
      
            int trainRealClass[][];
            int trainPrediction[][];
                
            trainRealClass = new int[trainData.length][1];
            trainPrediction = new int[trainData.length][1];	
           
             int nClases = DE.getPosibleValuesOfOutput().size();
                 
           //Working on training
            for (int i=0; i<trainingDataSet.size(); i++) {
                 trainRealClass[i][0] = (int) trainingDataSet.get(i).getOutput(0);
                 trainPrediction[i][0] = evaluate(trainingDataSet.get(i).getInputs(),DE.prototypeSetTodouble(), nClases, DE.getClases(), 1);
            }
            

            
            writeOutput(outFile[0], trainRealClass, trainPrediction); //, entradas, salida, relation);
            
            int realClass[][] = new int[testData.length][1];
            int prediction[][] = new int[testData.length][1];	
    	
    			
            for (int i=0; i<realClass.length; i++) {
            	realClass[i][0] = (int) testDataSet.get(i).getOutput(0);
            	prediction[i][0]= evaluate(testDataSet.get(i).getInputs(),DE.prototypeSetTodouble(), nClases, DE.getClases(), 1);
            }
                
            writeOutput(outFile[1], realClass, prediction); //,  entradas, salida, relation);
        
       
            String Reduction = new String("Reduccion = ");
            Reduction = "Reduccion = "+ this.getReduction();
            
        	Files.writeFile (outFile[2], Reduction);
        	
        
		Timer.setModelTime();
		Timer.resetTime();
	}
	
	public static void evaluationSpent(){
		evals++;
	}
	private void updateCollaborators(){
		
		for(int i=0; i<populations.length;i++){
			
			collaborators[i]=populations[i].getBest();

		}
	}
	
	public static Chromosome getCollaborator(int pop){
		
		return collaborators[pop].clone();
	}
	
	public static int getNPops(){
		
		return collaborators.length;
	}
	
	public static int getTrainSize(){
		
		return trainSize;
	}
	
	public int getClases(){
		
		return nClasses;
	}
	
	private void generateFinalReferenceSet(){
		
		
		int pointer=0;
		
		arrayIS= new int [CoIS.getTrainSize()];
		//generate IS array
		for(int i=0;i<collaborators.length;i++){
			for(int j=0;j<collaborators[i].getBody().length;j++){
				arrayIS[pointer]=collaborators[i].get(j);
				pointer++;
			}
		}
		
		KNN.setIS(arrayIS);

	}
	
	public int [] classifyTraining(){
		
		int result []= new int [trainData.length];
		
		for(int i=0;i<trainData.length;i++){
			
			result[i]=KNN.classifyTestInstance(trainData[i]);
			
			//System.out.println("result = "+ result[i]+", trainouput= "+ trainOutput[i]);
		}
		
		Timer.setTrainingTime();
		Timer.resetTime();
		
		return result;
	}
	
	public int [] classifyTestSet(){
		
		int result []= new int [testData.length];

		for(int i=0;i<testData.length;i++){
			
			result[i]=KNN.classifyTestInstance(testData[i]);
			
		}
		
		Timer.setTestTime();
		Timer.resetTime();
		
		return result;
	}
	
	/** 
	 * Executes the classification of train dataset
	 * 
	 */	
	public void classifyTrain(){
		
		modelTime=Timer.getModelTime();	
		System.out.println(name+" "+ relation + " Model " + modelTime + "s");
		
		//Check  time		
		Timer.resetTime();
		
		int [] clasResult;
		
		trainRealClass = new int[trainData.length][1];
		trainPrediction = new int[trainData.length][1];			
		    
		clasResult=classifyTraining();
		
		for (int i=0; i<trainRealClass.length; i++) {
			trainRealClass[i][0]= trainOutput[i];
			trainPrediction[i][0]= clasResult[i];
		}
			
		trainingTime=Timer.getTrainingTime();
		
		//translation of output attribute for safety
		for(int i=0;i<trainRealClass.length;i++){
			trainRealClass[i][0]=Translator.safe2real(trainRealClass[i][0]);
		}
		
		//translation of output attribute for safety
		for(int i=0;i<trainPrediction.length;i++){
			trainPrediction[i][0]=Translator.safe2real(trainPrediction[i][0]);
		}
		
		//Writing results
		writeOutput(outFile[0], trainRealClass, trainPrediction);
		System.out.println(name+" "+ relation + " Training " + trainingTime + "s");
		
	}//end-method 
	
	/** 
	 * Executes the classification of test dataset
	 * 
	 */	
	public void classifyTest(){
		
		//Check  time		
		Timer.resetTime();
		
		int [] clasResult;
		
		testRealClass = new int[testData.length][1];
		testPrediction = new int[testData.length][1];			
		    
		clasResult=classifyTestSet();
		
		for (int i=0; i<testRealClass.length; i++) {
			testRealClass[i][0]= testOutput[i];
			testPrediction[i][0]= clasResult[i];
		}
			
		testTime=Timer.getTestTime();
		
		//translation of output attribute for safety
		for(int i=0;i<testRealClass.length;i++){
			testRealClass[i][0]=Translator.safe2real(testRealClass[i][0]);
		}
		
		//translation of output attribute for safety
		for(int i=0;i<testPrediction.length;i++){
			testPrediction[i][0]=Translator.safe2real(testPrediction[i][0]);
		}
		
		//Writing results
		writeOutput(outFile[1], testRealClass, testPrediction);
		System.out.println(name+" "+ relation + " Test " + testTime + "s");
		
	}//end-method 


	/**
	 * Prints the additional output file
	 */
	public void printExitValues(){

		String text="";		
		nClasses=oldNClasses;
		computeConfussionMatrixes();
		
		//Accuracy
		text+="Accuracy: "+getAccuracy()+"\n";
		text+="Accuracy (Training): "+getTrainAccuracy()+"\n";
		
		//Kappa
		text+="Kappa: "+getKappa()+"\n";
		text+="Kappa (Training): "+getTrainKappa()+"\n";
		
		//Accuracy
		text+="Reduction: "+getReduction()+"\n";
		
		//Unclassified
		text+="Unclassified instances: "+testUnclassified+"\n";
		text+="Unclassified instances (Training): "+trainUnclassified+"\n";	
		
		//Model time
		text+= "Model time: "+modelTime+" s\n";
		
		//Training time
		text+= "Training time: "+trainingTime+" s\n";
		
		//Test time
		text+= "Test time: "+testTime+" s\n";
		
		//Confusion matrix
		text+="Confussion Matrix:\n";
		for(int i=0;i<nClasses;i++){
			
			for(int j=0;j<nClasses;j++){
				text+=testConfMatrix[i][j]+"\t";
			}
			text+="\n";
		}
		text+="\n";
		
		text+="Training Confussion Matrix:\n";
		for(int i=0;i<nClasses;i++){
			
			for(int j=0;j<nClasses;j++){
				text+=trainConfMatrix[i][j]+"\t";
			}
			text+="\n";
		}
		text+="\n";

		//Finish additional output file
	//	Files.writeFile (outFile[2], text);
		
	}//end-method 
	
	/**
	 * Computes the confusion matrixes
	 * 
	 */
	private void computeConfussionMatrixes(){
		
		testConfMatrix= new int [nClasses][nClasses];
		trainConfMatrix= new int [nClasses][nClasses];
		
		testUnclassified=0;
		
		for(int i=0;i<nClasses;i++){
			Arrays.fill(testConfMatrix[i], 0);
		}
		
		for(int i=0;i<testPrediction.length;i++){
			if(testPrediction[i][0]==-1){
				testUnclassified++;
			}else{
				testConfMatrix[testPrediction[i][0]][testRealClass[i][0]]++;
			}
		}
		
		trainUnclassified=0;
		
		for(int i=0;i<nClasses;i++){
			Arrays.fill(trainConfMatrix[i], 0);
		}
		
		for(int i=0;i<trainPrediction.length;i++){
			if(trainPrediction[i][0]==-1){
				trainUnclassified++;
			}else{
				trainConfMatrix[trainPrediction[i][0]][trainRealClass[i][0]]++;
			}
		}
		
	}//end-method 
	
	private double getReduction(){
		
		double counter=0.0;
		
		for(int i=0;i<arrayIS.length;i++){
			if(arrayIS[i]==0){
				counter+=1.0;
			}
		}
		
		return counter/(double)arrayIS.length;
		
	}
	/**
	 * Computes the accuracy obtained on test set
	 * 
	 * @return Accuracy on test set
	 */
	private double getAccuracy(){
		
		double acc;
		int count=0;
		
		for(int i=0;i<nClasses;i++){			
			count+=testConfMatrix[i][i];
		}
		
		acc=((double)count/(double)test.getNumInstances());
		
		return acc;
		
	}//end-method 
	
	/**
	 * Computes the accuracy obtained on the training set
	 * 
	 * @return Accuracy on test set
	 */
	private double getTrainAccuracy(){
		
		double acc;
		int count=0;
		
		for(int i=0;i<nClasses;i++){			
			count+=trainConfMatrix[i][i];
		}
		
		acc=((double)count/(double)train.getNumInstances());
		
		return acc;
		
	}//end-method 
	
	/**
	 * Computes the Kappa obtained on test set
	 * 
	 * @return Kappa on test set
	 */	
	private double getKappa(){
		
		double kappa;
		double agreement,expected;
		int count,count2;
		double prob1,prob2;
		
		count=0;
		for(int i=0;i<nClasses;i++){			
			count+=testConfMatrix[i][i];
		}
		
		agreement=((double)count/(double)test.getNumInstances());
		
		expected=0.0;
		
		for(int i=0;i<nClasses;i++){			
			
			count=0;
			count2=0;
			
			for(int j=0;j<nClasses;j++){
				count+=testConfMatrix[i][j];
				count2+=testConfMatrix[j][i];
			}
			
			prob1=((double)count/(double)test.getNumInstances());
			prob2=((double)count2/(double)test.getNumInstances());
			
			expected+=(prob1*prob2);
		}

		kappa=(agreement-expected)/(1.0-expected);
		
		return kappa;
		
	}//end-method 

	/**
	 * Computes the Kappa obtained on test set
	 * 
	 * @return Kappa on test set
	 */	
	private double getTrainKappa(){
		
		double kappa;
		double agreement,expected;
		int count,count2;
		double prob1,prob2;
		
		count=0;
		for(int i=0;i<nClasses;i++){			
			count+=trainConfMatrix[i][i];
		}
		
		agreement=((double)count/(double)train.getNumInstances());
		
		expected=0.0;
		
		for(int i=0;i<nClasses;i++){			
			
			count=0;
			count2=0;
			
			for(int j=0;j<nClasses;j++){
				count+=trainConfMatrix[i][j];
				count2+=trainConfMatrix[j][i];
			}
			
			prob1=((double)count/(double)train.getNumInstances());
			prob2=((double)count2/(double)train.getNumInstances());
			
			expected+=(prob1*prob2);
		}

		kappa=(agreement-expected)/(1.0-expected);
		
		return kappa;
		
	}//end-method 
	
	/**
	 * Prints output files.
	 * 
	 * @param filename Name of output file
	 * @param realClass Real output of instances
	 * @param prediction Predicted output for instances
	 */
	private void writeOutput(String filename, int [][] realClass, int [][] prediction) {
	
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
    
    
	
	
	
	// isaak code
	
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

	  
		protected static double distance(Prototype instance1, Prototype instance2){
			
			double length=0.0;

			for (int i=0; i<instance1.numberOfInputs(); i++) {
				length += (instance1.getInput(i)-instance2.getInput(i))*(instance1.getInput(i)-instance2.getInput(i));
			}
				
			length = Math.sqrt(length); 
					
			return length;
			
		} //end-method
		
	  
	  
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
	           currDist = distance(pi,current);
	          // System.out.println(currDist);
	          
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
	  
	  
	  
	  
	/** Main method */
	  
	  public PrototypeSet reduceSet(PrototypeSet initial){

		  
	 // decide the structure of the particle.

		  //trainingDataSet.print();
		  
		  PrototypeSet randomize=new PrototypeSet(initial.clone()) ;
	
		  this.numberOfClass = trainingDataSet.getPosibleValuesOfOutput().size();
		  
		  // Initial Fitness:
		  PrototypeSet nominalPopulation = new PrototypeSet();
	      nominalPopulation.formatear(randomize);
	      
	      if(randomize.size()>=2){
	       System.err.println("\nInitial % de acierto en training Nominal " + classficationAccuracy1NN(nominalPopulation,trainingDataSet) );
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
						
						  for(int z=0; z< population[i][0].size(); z++){
							  population[i][j].add(trainingDataSet.getFromClass(i).getRandom());
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
		
		
} //end-class 
