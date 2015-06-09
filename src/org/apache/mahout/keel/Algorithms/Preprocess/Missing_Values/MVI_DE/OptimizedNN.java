package org.apache.mahout.keel.Algorithms.Preprocess.Missing_Values.MVI_DE;

import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;


public class OptimizedNN{
	
	double[] stdDev;
	double[][][] nominalDistance;
	
	// optimization of the accuracy computation
	public double[] minDistance;
	public int[] nearestNeigbor;
	public int[] prediction;	

	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************

	public OptimizedNN(){
		
		minDistance = new double[Parameters.numInstancesTRA];
		nearestNeigbor = new int[Parameters.numInstancesTRA];
		prediction = new int[Parameters.numInstancesTRA];
}

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************

	public double getInitialAccuracy(){
		
		computeHVDM();
		
		double correct = 0;
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			
			evaluate(i);
			
			if(Parameters.classTRA[i] == prediction[i])
				correct++;
		}
		
		return correct/Parameters.numInstancesTRA;
	}
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************

	public double getOptimizedAccuracy(int example, double previousACC){
		
		// el ejemplo esta movido, luego es necesario re-calcular el HVDM
		if(Parameters.iterativeHVDM)
			computeHVDM();
				
		// 1) copio en un nuevo vector los resultados de prediction, minDist and nearestneigbors
		int[] pre_aux = new int[Parameters.numInstancesTRA];
		double[] min_aux = new double[Parameters.numInstancesTRA];
		int[] near_aux = new int[Parameters.numInstancesTRA];
		System.arraycopy(prediction, 0, pre_aux, 0, Parameters.numInstancesTRA);
		System.arraycopy(minDistance, 0, min_aux, 0, Parameters.numInstancesTRA);
		System.arraycopy(nearestNeigbor, 0, near_aux, 0, Parameters.numInstancesTRA);
		
		// 2) calcular la distancia de todos los ejemplos a example
		double[] dx = new double[Parameters.numInstancesTRA];
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i)
			if(i != example)
				dx[i] = distance(example,i);
		dx[example] = Double.MAX_VALUE;
		
		
		// 1) fijo la clase de x
		double min = Double.MAX_VALUE;
		int pos = -1;
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			if(dx[i] < min){
				min = dx[i];
				pos = i;
			}
		}
		
		prediction[example] = Parameters.classTRA[pos];
		minDistance[example] = min;
		nearestNeigbor[example] = pos;
		
		// 2) Los ejemplos que fueron clasificados por X, se vuelve a calcular su clase con los nuevos datos
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			if(nearestNeigbor[i] == example){
				evaluate(i);
			}
		}
		
		// 3) comparar minimas distancias anteriores con las nuevas a x. Si alguna es menor, asigno clase de x a ese ejemplo
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			if(dx[i] < minDistance[i] && i != example){
				prediction[i] = Parameters.classTRA[example];
				minDistance[i] = dx[i];
				nearestNeigbor[i] = example;
			}
		}
		

		// 6) cuento aciertos
		double correct = 0;
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
						
			if(Parameters.classTRA[i] == prediction[i])
				correct++;
		}
		
		double newACC = correct/Parameters.numInstancesTRA;

		// si no mejoro, vuelvo a valores originales
		if(newACC < previousACC){
			System.arraycopy(pre_aux, 0, prediction, 0, Parameters.numInstancesTRA);
			System.arraycopy(min_aux, 0, minDistance, 0, Parameters.numInstancesTRA);
			System.arraycopy(near_aux, 0, nearestNeigbor, 0, Parameters.numInstancesTRA);
		}
		
		return newACC;
	}	

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public double distance (int ex1, int ex2){

		double result = 0;
		
		for (int i = 0 ; i < Parameters.numAttributes ; i++){
			
			if (Parameters.attributeType[i] == Attribute.NOMINAL) {
				result += nominalDistance[i][Parameters.nominalIntTRA[ex1][i]][Parameters.nominalIntTRA[ex2][i]];
			}
			
            if (Parameters.attributeType[i] == Attribute.INTEGER) {
                if(stdDev[i] == 0)
                        result += 0;
                else                                
                        result += Math.abs(Parameters.integerValueTRA[ex1][i]-Parameters.integerValueTRA[ex2][i]) / (4*stdDev[i]);
	        }
	        
	        if (Parameters.attributeType[i] == Attribute.REAL) {
	                if(stdDev[i] == 0)
	                        result += 0;
	                else
	                        result += Math.abs(Parameters.instanceTRA[ex1][i]-Parameters.instanceTRA[ex2][i]) / (4*stdDev[i]);
	        }
		}
	
		result = Math.sqrt(result);       	
	
		return result;  
}
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public void evaluate(int example){
		
		double minDist = Double.MAX_VALUE;
		int nearestN = -1;

	
		for (int i = 0 ; i < Parameters.numInstancesTRA ; i++) {
		
		    double dist = distance(i, example);

			if (dist > 0.0){ //leave-one-out
			
				//see if it's nearer than our previous selected neighbors				
				if (dist < minDist) {
					
					minDist = dist;
					nearestN = i;
				}
				
			}
		}
		
		minDistance[example] = minDist;
		nearestNeigbor[example] = nearestN;
		prediction[example] = Parameters.classTRA[nearestN];
	}

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************

	public void computeHVDM(){
	
		double VDM, Nax, Nay, Naxc, Nayc, media, SD;

		stdDev = new double[Parameters.numAttributes];
		nominalDistance = new double[Parameters.numAttributes][][];
		
		for (int i = 0 ; i < Parameters.numAttributes ; i++){
			
			if (Parameters.attributeType[i] == Attribute.NOMINAL){
				
				nominalDistance[i] = new double[Attributes.getInputAttribute(i).getNumNominalValues()][Attributes.getInputAttribute(i).getNumNominalValues()];
                 for (int j = 0 ; j < Attributes.getInputAttribute(i).getNumNominalValues(); j++)
                	 nominalDistance[i][j][j] = 0.0;
                 
                 
                 for (int j=0; j<Attributes.getInputAttribute(i).getNumNominalValues(); j++) {
                	 
                	 for (int l=j+1; l<Attributes.getInputAttribute(i).getNumNominalValues(); l++) {
                     
                		 VDM = 0.0;
                         Nax = Nay = 0;
                         
                         for (int m=0; m<Parameters.numInstancesTRA; m++){
                        	 
                        	 if(Parameters.iterativeHVDM){
                        		 if (Parameters.nominalIntTRA[m][i] == j) 
                        			 Nax++;
                        	 
                        		 if (Parameters.nominalIntTRA[m][i] == l) 
                        			 Nay++;
                        	 }
                        	 
                        	 else{
                        		 if (Parameters.nominalIntTRA[m][i] == j && !Parameters.missingTRA[m][i]) 
                        			 Nax++;
                        	 
                        		 if (Parameters.nominalIntTRA[m][i] == l && !Parameters.missingTRA[m][i]) 
                        			 Nay++;
                        	 }
                         }
                                          
                         
                         for (int m=0; m<Parameters.numClasses; m++){
                        	 
                        	 Naxc = Nayc = 0;
                             
                        	 for (int n=0; n<Parameters.numInstancesTRA; n++) {
                        		 
                        		 if(Parameters.iterativeHVDM){
                            		 if (Parameters.nominalIntTRA[n][i] == j && Parameters.classTRA[n] == m) 
                            			 Naxc++;
                                                
                                      if (Parameters.nominalIntTRA[n][i] == l && Parameters.classTRA[n] == m) 
                                    	  Nayc++;
                        		 }
                        		 
                        		 else{
                            		 if (Parameters.nominalIntTRA[n][i] == j && Parameters.classTRA[n] == m && !Parameters.missingTRA[n][i]) 
                            			 Naxc++;
                                                
                                      if (Parameters.nominalIntTRA[n][i] == l && Parameters.classTRA[n] == m && !Parameters.missingTRA[n][i]) 
                                    	  Nayc++; 
                        		 }

                        	 }
                        	 
                        	 
                        	 VDM += (((double)Naxc / (double)Nax) - ((double)Nayc / (double)Nay)) * (((double)Naxc / (double)Nax) - ((double)Nayc / (double)Nay));
                         }
                                    
                         
                         nominalDistance[i][j][l] = Math.sqrt(VDM);
                         nominalDistance[i][l][j] = Math.sqrt(VDM);
                	 }        
                 }
			}
			
			else{
				
				int cont_ins = 0;
				media = 0; 
				SD = 0;
                    
				for (int j = 0 ; j < Parameters.numInstancesTRA ; j++){
					
					if(Parameters.iterativeHVDM){
						cont_ins++;
						media += Parameters.instanceTRA[j][i];
						SD += Parameters.instanceTRA[j][i]*Parameters.instanceTRA[j][i];						
					}
					
					else{
						if(!Parameters.missingTRA[j][i]){
							cont_ins++;
							media += Parameters.instanceTRA[j][i];
							SD += Parameters.instanceTRA[j][i]*Parameters.instanceTRA[j][i];						
						}
					}
					
				}
                    
				media /= (double)cont_ins;  
				stdDev[i] = Math.sqrt((SD/((double)cont_ins)) - (media*media));
			}

		}
	}

}