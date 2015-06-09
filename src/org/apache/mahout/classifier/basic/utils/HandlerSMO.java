package org.apache.mahout.classifier.basic.utils;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Algorithms.SVM.SMO.SMO;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceSet;

public class HandlerSMO {
		
	private int[][] predictions;
	ArrayList<Integer> predicciones;
	private double[][] probabilities;
	private String algSufix = "SMO";
	
    public int numPartitions;
    public int numInstances;
    public int numClasses;
    public String trainInputFile;
    public String testInputFile;
    public String seed;
    
	
	public HandlerSMO(int clases, int instances, String Seed){
	      this.numPartitions = 1;
	      this.trainInputFile= "train1.dat";
	      this.testInputFile = "test1.dat";
	      this.numClasses = clases;
	      this.numInstances = instances;
	      this.seed = Seed;
	      
	}
	
	public HandlerSMO(InstanceSet train, InstanceSet test, int clases, String Seed, String cabecera) throws Exception{

	      this.numPartitions = 1;
	      this.trainInputFile= "train1.dat";
	      this.testInputFile = "test1.dat";
	      this.numClasses = clases;
	      this.numInstances = test.getNumInstances();
	      this.seed = Seed;
		
		// crear ficheros de configuracion
		createConfigurationFiles();

		for(int i = 0 ; i < this.numPartitions ; ++i){

	        String[] argumentos = new String[1];
	        argumentos[0] = "config_" + algSufix + "_" + (i+1) + ".txt";
	        
			SMO model = new SMO (argumentos[0]);

			model.runModel(train, test, cabecera);
			probabilities = model.probabilities.clone();
			
			//leer las instancias de cada particion		
			predictions = new int[this.numPartitions][this.numInstances];
			
			predictions[0]=model.predictions;
			predicciones=model.predicciones;

		}
		
		// borrar ficheros de configuracion
		for(int i = 0 ; i < this.numPartitions ; ++i){
			File f = new File("config_" + algSufix + "_" + (i+1) + ".txt");
			f.delete();
		}
		

		/*
		for(int i = 0 ; i < this.numPartitions ; ++i){
			
			BufferedReader fE = new BufferedReader(new FileReader("test_" + algSufix + "_" + (i+1) + ".dat"));
			
			while(!fE.readLine().contains("@data"));
			
			// guardo las clases predichas
			for(int q = 0 ; q < this.numInstances ; ++q){
				
				String linea = fE.readLine();
				String salida = linea.split(" ")[1];
				
				int claseInt = 0;
				boolean seguir = true;
				for(int sa = 0 ; sa < this.numClasses && seguir ; ++sa){
					//System.out.println(salida);
					//ystem.out.println(Attributes.getOutputAttribute(0));
					//System.out.println(sa);
					if(Attributes.getOutputAttribute(0).getNominalValue(sa).equals(salida)){
						claseInt = sa;
						seguir = false;
					}
				}
			
				
				predictions[i][q] = claseInt;
			}
			
			fE.close();
		}
		*/
		
	}
	
	public HandlerSMO(InstanceSet train, FileSystem fs, Path test, int clases, String Seed, String cabecera, Context context, int partition) throws Exception{

	      this.numPartitions = 1;
	      this.trainInputFile= "train1.dat";
	      this.testInputFile = "test1.dat";
	      this.numClasses = clases;
	     // this.numInstances = test.getNumInstances();
	      this.seed = Seed;
		
	      Utils.readHeader(cabecera);
		// crear ficheros de configuracion
		createConfigurationFiles();

		
	        String[] argumentos = new String[1];
	        argumentos[0] = "config_" + algSufix + "_" + 1 + ".txt";
	        
			SMO model = new SMO (argumentos[0]);

			model.runModel(train, fs, test,cabecera, context,partition);
			
			
			probabilities = null; //model.probabilities.clone();
			
			predicciones=model.predicciones;

			// borrar ficheros de configuracion
			File f = new File("config_" + algSufix + "_" + 1 + ".txt");
			f.delete();
	
		
	}
	
	
	public void generateFiles() throws Exception{
		
		// crear ficheros de configuracion
		createConfigurationFiles();
		
		// ejecutar el metodo
		for(int i = 0 ; i < this.numPartitions ; ++i){
	        Attributes.clearAll();
	        String[] argumentos = new String[1];
	        argumentos[0] = "config_" + algSufix + "_" + (i+1) + ".txt";
	        org.apache.mahout.keel.Algorithms.SVM.SMO.Main.main(argumentos);
		}
		
		// borrar ficheros de configuracion
		for(int i = 0 ; i < this.numPartitions ; ++i){
			File f = new File("config_" + algSufix + "_" + (i+1) + ".txt");
			f.delete();
		}
		
		//leer las instancias de cada particion		
		predictions = new int[this.numPartitions][this.numInstances];
		for(int i = 0 ; i < this.numPartitions ; ++i){
			
			BufferedReader fE = new BufferedReader(new FileReader("test_" + algSufix + "_" + (i+1) + ".dat"));
			
			while(!fE.readLine().contains("@data"));
			
			// guardo las clases predichas
			for(int q = 0 ; q < this.numInstances ; ++q){
				
				String linea = fE.readLine();
				String salida = linea.split(" ")[1];
				
				int claseInt = 0;
				boolean seguir = true;
				for(int sa = 0 ; sa < this.numClasses && seguir ; ++sa){
					
					System.out.println(salida);
					if(Attributes.getOutputAttribute(0).getNominalValue(sa).equals(salida)){
						System.out.println(salida);
						claseInt = sa;
						seguir = false;
					}
				}
				
			
				
				predictions[i][q] = claseInt;
			}
			
			fE.close();
		}
		
		//
		Attributes.clearAll();
		try {
			InstanceSet finalIS = new InstanceSet();
			finalIS.readSet(this.trainInputFile, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public void deleteFiles(){
		
		for(int i = 0 ; i < this.numPartitions ; ++i){
			File f = new File("train_" + algSufix + "_" + (i+1) + ".dat");
			f.delete();
			
			f = new File("test_" + algSufix + "_" + (i+1) + ".dat");
			f.delete();
		}
		
	}
	
	public int[] getPredictions(int part){
		return predictions[part];
	}
	
	public ArrayList<Integer> getPredicciones(){
		return predicciones;
	}
	
	
	public double[][] getProbabilities(){
		return probabilities;
	}

//*******************************************************************************************************************************
  	
  	private void createConfigurationFiles() throws IOException{
  		
  		for(int i = 0 ; i < this.numPartitions ; ++i){
  			BufferedWriter bf = new BufferedWriter(new FileWriter("config_" + algSufix + "_"+(i+1)+".txt"));
  			String cad = "";
  			cad += "algorithm = " + algSufix + "\n";
  			cad += "inputData = \""+this.trainInputFile+"\""+" \""+this.trainInputFile+"\""+" \""+ this.testInputFile + "\"\n";
  			cad += "outputData = \"train_" + algSufix + "_" + (i+1) + ".dat\" \"test_" + algSufix + "_" + (i+1) + ".dat\" \"test2_" + algSufix + "_" + (i+1) + ".dat\"\n\n";
  			cad += "seed = " + this.seed + "\n";
  			cad += "C = 100.0\n";
  			cad += "toleranceParameter = 0.001\n";
  			cad += "epsilon = 1.0E-12\n";
  			cad += "RBFKernel_gamma = 0.01\n";
  			cad += "-Normalized-PolyKernel_exponent = 1\n";
  			cad += "-Normalized-PolyKernel_useLowerOrder = False\n";
  			cad += "PukKernel_omega = 1.0\n";
  			cad += "PukKernel_sigma = 1.0\n";
  			cad += "StringKernel_lambda = 0.5\n";
  			cad += "StringKernel_subsequenceLength = 3\n";
  			cad += "StringKernel_maxSubsequenceLength = 9\n";
  			cad += "StringKernel_normalize = False\n";
  			cad += "StringKernel_pruning = None\n";
  			//cad += "KERNELtype = Puk\n";
  			cad += "KERNELtype = RBFKernel\n";
  			cad += "FitLogisticModels = False\n"; // True
  			cad += "ConvertNominalAttributesToBinary = False\n";
  			cad += "PreprocessType = Normalize"; // 
  			
  			
  			//System.out.println(cad);
  			bf.write(cad);
  			bf.close();
  		}
  		
  	}

	
}
