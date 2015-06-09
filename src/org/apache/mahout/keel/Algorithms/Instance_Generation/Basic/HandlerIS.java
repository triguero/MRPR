package org.apache.mahout.keel.Algorithms.Instance_Generation.Basic;



import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.mahout.classifier.pg.data.*;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.Reconsistent_Imb.Reconsistent;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.SSMA_Imb.SSMA;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.SSMA_Imb.SSMA_window;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.ENNTh_Imb.ENNTh;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.FCNN.*;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.HMNEI_Imb.HMNEI;
import org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.DROP3.*;

public class HandlerIS {
		
	private int[][] predictions;
	private String algSufix = "AllKNN";
	
	public PrototypeSet reduceSet;
	
	public HandlerIS(){
	}
	
	public HandlerIS(String nombre){
		this.algSufix = nombre;
	}
	
	
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
	  
	public PrototypeSet reduce(Data data, Context context) throws IOException{
		PrototypeSet trainPG=new PrototypeSet(data);
	    context.progress();
	    
		createConfigurationFiles();
        String[] argumentos = new String[1];
        argumentos[0] = "NOFILE";
	    
	    if(this.algSufix.equalsIgnoreCase("FCNN")){
			FCNN model;
			
			model= new FCNN(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(); 
        } else if(this.algSufix.equalsIgnoreCase("DROP3")){
			DROP3 model;
			
			model= new DROP3(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(); 
        } else if(this.algSufix.equalsIgnoreCase("SSMAImb")){
			SSMA model;
			
			model= new SSMA(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(context); 
        } else if(this.algSufix.equalsIgnoreCase("SSMAImb-W")){
			SSMA_window model;
			
			model= new SSMA_window(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(context); 
        } else if(this.algSufix.equalsIgnoreCase("ENNThImb")){
			ENNTh model;
			
			model= new ENNTh(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(context); 
        } else if(this.algSufix.equalsIgnoreCase("HMNEIImb")){
			HMNEI model;
			
			model= new HMNEI(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(context); 
        } else if(this.algSufix.equalsIgnoreCase("ReconsImb")){
			Reconsistent model;
			
			model= new Reconsistent(argumentos[0], trainPG.toInstanceSet());
			model.ejecutar(context); 
        }
	    
	    PrototypeSet training = readPrototypeSet("salida.dat");		
	    return training;
	}
	
	/*
	public int[] ejecutar(InstanceSet train) throws Exception{
		
		
			
		// crear ficheros de configuracion
		createConfigurationFiles();
		
		
		
		
		// ejecutar el metodo
		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
	       // Attributes.clearAll();
	        String[] argumentos = new String[1];
	        argumentos[0] = "config_" + algSufix + "_" + (i+1) + ".txt";
		
	        if(this.algSufix.equalsIgnoreCase("AllKNN")){
				AllKNN allKNN;
		
    			allKNN= new AllKNN(argumentos[0],train);
    			train.print();
				allKNN.ejecutar(); 

	        }else if(this.algSufix.equalsIgnoreCase("MoCS")){
				ModelCS model;
				
				model= new ModelCS(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("MENN")){
				MENN model;
				
				model= new MENN(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("NCNEdit")){
				NCNEdit model;
				
				model= new NCNEdit(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENN")){
				ENN model;
				
				model= new ENN(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("Multiedit")){
				Multiedit model;
				
				model= new Multiedit(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("RNGE")){
				RNG model;
				
				model= new RNG(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENRBF")){
				ENRBF model;
				
				model= new ENRBF(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENNTh")){
				ENNTh model;
				
				model= new ENNTh(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("FCNN")){
				FCNN model;
				
				model= new FCNN(argumentos[0],train);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("DROP3")){
				DROP3 model;
				
				model= new DROP3(argumentos[0],train);
				model.ejecutar(); 
	        }
	        
		}
		
		// borrar ficheros de configuracion
		
		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
			File f = new File("config_" + algSufix + "_" + (i+1) + ".txt");
			f.delete();
		}
		
		// Determinar que instancias son ruidosas.

		PrototypeSet mojontron = new PrototypeSet(train);
		
		//mojontron.print();
		
		InstanceSet IS = new InstanceSet();
		InstanceSet finalIS = new InstanceSet();
		
		
	
		try {
			Attributes.clearAll();
			finalIS.readSet("train_" + algSufix + "_1.dat" , true);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		
		Instance[] instances, instances2;
		instances = train.getInstances();
		instances2 = finalIS.getInstances();
		
		System.out.println(instances.length);
		System.out.println(instances2.length);
		int[] ruidosas = new int[instances.length-instances2.length]; // number of noisy instances.
		int cont =0;
		
		for(int i=0; i<instances.length; i++){
			
			//Cheking if there is this instance in the cleaning data set.
			boolean esta = false;
			for(int j=0; j<instances2.length && !esta; j++){
						
				if(instances[i].toString().equalsIgnoreCase(instances2[j].toString())){
					esta = true;
					System.out.print("esta");
				}
				
			}
			
			if(!esta && ruidosas.length>0){
				System.out.println("ENTRO");
				//System.out.println(ruidosas.length);
				ruidosas[cont]=i;
			//	System.out.println(i);
				cont++;
			}
			
		}
		              
		return ruidosas;
	}
	
	
	public int[] generateFiles() throws Exception{
		
		// crear ficheros de configuracion
		createConfigurationFiles();
		
		// ejecutar el metodo
		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
	        Attributes.clearAll();
	        String[] argumentos = new String[1];
	        argumentos[0] = "config_" + algSufix + "_" + (i+1) + ".txt";
		
	        if(this.algSufix.equalsIgnoreCase("AllKNN")){
				AllKNN allKNN;
		
				allKNN= new AllKNN(argumentos[0]);
				allKNN.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("MoCS")){
				ModelCS model;
				
				model= new ModelCS(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("MENN")){
				MENN model;
				
				model= new MENN(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("NCNEdit")){
				NCNEdit model;
				
				model= new NCNEdit(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENN")){
				ENN model;
				
				model= new ENN(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("Multiedit")){
				Multiedit model;
				
				model= new Multiedit(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("RNGE")){
				RNG model;
				
				model= new RNG(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENRBF")){
				ENRBF model;
				
				model= new ENRBF(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("ENNTh")){
				ENNTh model;
				
				model= new ENNTh(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("FCNN")){
				FCNN model;
				
				model= new FCNN(argumentos[0]);
				model.ejecutar(); 
	        }else if(this.algSufix.equalsIgnoreCase("DROP3")){
				DROP3 model;
				
				model= new DROP3(argumentos[0]);
				model.ejecutar(); 
	        }
	        
		}
		
		// borrar ficheros de configuracion
		
		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
			File f = new File("config_" + algSufix + "_" + (i+1) + ".txt");
			f.delete();
		}
		
		// Determinar que instancias son ruidosas.
		

		
		InstanceSet IS = new InstanceSet();
		InstanceSet finalIS = new InstanceSet();
		
	
		try {
			Attributes.clearAll();
			IS.readSet(ParametersSMO.trainInputFile, true);
			Attributes.clearAll();
			finalIS.readSet("train_" + algSufix + "_1.dat" , true);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Instance[] instances, instances2;
		instances = IS.getInstances();
		instances2 = finalIS.getInstances();
		
		//System.out.println(instances.length);
		//System.out.println(instances2.length);
		int[] ruidosas = new int[instances.length-instances2.length]; // number of noisy instances.
		int cont =0;
		
		for(int i=0; i<instances.length; i++){
			
			//Cheking if there is this instance in the cleaning data set.
			boolean esta = false;
			for(int j=0; j<instances2.length && !esta; j++){
						
				if(instances[i].toString().equalsIgnoreCase(instances2[j].toString())){
					esta = true;
				}
				
			}
			
			if(!esta){
				ruidosas[cont]=i;
			//	System.out.println(i);
				cont++;
			}
			
		}
		              
		return ruidosas;
	}
	
	public void deleteFiles(){
		
		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
			File f = new File("train_" + algSufix + "_" + (i+1) + ".dat");
			f.delete();
			
			f = new File("test_" + algSufix + "_" + (i+1) + ".dat");
			f.delete();
		}
		
	}
	
	public int[] getPredictions(int part){
		return predictions[part];
	}
*/
//*******************************************************************************************************************************
  	
  	private void createConfigurationFiles() throws IOException{
  		
  		for(int i = 0 ; i < ParametersSMO.numPartitions ; ++i){
  			BufferedWriter bf = new BufferedWriter(new FileWriter("config_" + algSufix + "_"+(i+1)+".txt"));
  			String cad = "";
  			cad += "algorithm = " + algSufix + "\n";
  			cad += "inputData = \"" + ParametersSMO.trainInputFile + "\" " +  "\""+  ParametersSMO.trainInputFile + "\" " +   "\"" +ParametersSMO.trainInputFile+ "\"\n";
  			cad += "outputData = \"train_" + algSufix + "_" + (i+1) + ".dat\" \"test_" + algSufix + "_" + (i+1) + ".dat\"\n\n";
  			//cad += "seed = " + ParametersSMO.seed + "\n";
  		
  			if(algSufix.equalsIgnoreCase("RNGE")){
  				cad += "Order of the Graph= 1st_order\n";
  				cad += "Type of Selection= Edition\n";

  			}else if(algSufix.equalsIgnoreCase("ENRBF")){
  				cad += "Sigma = 1\n";
  				cad += "Alpha= 1\n";
  			}else if(algSufix.equalsIgnoreCase("FCNN")||algSufix.equalsIgnoreCase("DROP3")){
  				cad += "number of Neighbors= 3\n";
  			}else if(algSufix.equalsIgnoreCase("SSMAImb")){ // SARAH
  				cad += "population size= 30\n";
  				cad += "number of evaluations= 10000\n";
  				cad += "cross probability per bit= 0.5\n";
  				cad += "mutation probability per bit= 0.001\n";
  				cad += "number of neighbors= 1\n";
  				cad += "distance function= euclidean\n";
  				cad += "penalization factor= 0.2\n";
  				cad += "mu= 50\n";
  				cad += "use fscore instead of mean= no\n";	
  			}else if(algSufix.equalsIgnoreCase("SSMAImb-W")){ // SARAH
  				cad += "population size= 30\n";
  				cad += "number of evaluations= 10000\n";
  				cad += "cross probability per bit= 0.5\n";
  				cad += "mutation probability per bit= 0.001\n";
  				cad += "number of neighbors= 1\n";
  				cad += "distance function= euclidean\n";
  				cad += "penalization factor= 0.2\n";
  				cad += "mu= 50\n";
  				cad += "use fscore instead of mean= no\n";	
  				cad += "nWindowPos= 1\n";
  				cad += "nWindowNeg= -1\n";
  			}else if(algSufix.equalsIgnoreCase("ENNThImb")){ // SARAH
  				cad += "number of neighbors= 3\n";
  				cad += "noise threshold= 0.7\n";  				
  				cad += "distance function= euclidean\n";
  				cad += "which version= 5\n";
  			}else if(algSufix.equalsIgnoreCase("HMNEIImb")){ // SARAH
  				cad += "epsilon= 0.1\n";
  				cad += "distance function= euclidean\n";
  				cad += "which version= 1\n";
  			}else if(algSufix.equalsIgnoreCase("ReconsImb")){ // SARAH
  				cad += "number of neighbors= 1\n";
  				cad += "distance function= euclidean\n";
  			} else if(algSufix.equalsIgnoreCase("ENNTh")){
  				cad += "number of Neighbors= 3\n";
  				cad += "noise thresohold= 0.7\n";
  			}else{
	  			if(algSufix.equalsIgnoreCase("Multiedit")){
	  				cad += "seed = " + ParametersSMO.seed + "\n";
	  				cad += "number of Neighbors= 1\n";
	  				cad += "number of Subblocks= 3\n";
	  			}else{
	  				cad += "number of Neighbors= 7\n";
	  			}

  			}
  			
  			cad += "Distance Function = Euclidean\n";
  			
  			bf.write(cad);
  			bf.close();
  		}
  		
  	}

	
}
