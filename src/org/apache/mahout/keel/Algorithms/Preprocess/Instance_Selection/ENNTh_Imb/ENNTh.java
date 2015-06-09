package org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.ENNTh_Imb;


import java.util.ArrayList;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Algorithms.Preprocess.Basic.*;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.*;
import org.apache.hadoop.mapreduce.Mapper.Context;

import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.Collections;


public class ENNTh extends Metodo {
	
	/*Own parameters of the algorithm*/
	  private int k;
	  private double mu;
	  private int version;

	  public ENNTh (String ficheroScript) {
	    super (ficheroScript);
	  }
	  
	  public ENNTh (String ficheroScript, InstanceSet train) {
	    super (ficheroScript, train);
	 }

	  
	  public void ejecutar (Context context) {
		  
		  System.out.println("I am executing ENNTh_Imb");

	    int i, j, l;
	    int nClases;
	    int claseObt;
	    boolean marcas[];
	    int nSel;    
	    double conjS[][];
	    double conjR[][];
	    int conjN[][];
	    boolean conjM[][];
	    int clasesS[];
	    int vecinos[];
	    double prob[];
	    double sumProb;
	    double maxProb;
	    int pos;

	    long tiempo = System.currentTimeMillis();

	    /*Inicialization of the flagged instances vector for a posterior copy*/
	    marcas = new boolean[datosTrain.length];
	    for (i=0; i<datosTrain.length; i++){
	      marcas[i] = true;
	    }
	    nSel = datosTrain.length;

	    /*Getting the number of differents classes*/
	    nClases = 0;
	    for (i=0; i<clasesTrain.length; i++)
	      if (clasesTrain[i] > nClases)
	        nClases = clasesTrain[i];
	    nClases++;
	    
	    /*
	     * Determine minority and majority class (binary) and IR 
	     * of the original set.
	     */
	    int[] classDistr = new int[nClases];
	    for (i = 0; i < clasesTrain.length; i++){
	        classDistr[clasesTrain[i]]++;
	    }
	    int posClass;
	    if(classDistr[0] < classDistr[1]){
	        posClass = 0;
	    } else {
	        posClass = 1;
	    }
	    double origIR = (double) classDistr[posClass ^ 1]/ classDistr[posClass];
	    
	    context.progress();
	    
	    vecinos = new int[k];
	    double[][] priors = new double[datosTrain.length][nClases];
	    
	    // Determine the probabilities p_+ and p_- for all instances
	    if(version <= 3){
	        for (i=0; i<datosTrain.length; i++) {    
	            if(clasesTrain[i] == posClass){
	                priors[i][posClass] = 1.0;
	                priors[i][posClass ^ 1] = 0.0;
	            } else {
	              KNN.evaluacionKNN2(k, datosTrain, realTrain, nominalTrain, nulosTrain, 
	                      clasesTrain, datosTrain[i], realTrain[i], nominalTrain[i], 
	                      nulosTrain[i], nClases, distanceEu, vecinos);
	              int[] ks = new int[nClases];
	              for(j = 0; j < vecinos.length; j++){
	                  if(vecinos[j] >= 0){
	                      ks[clasesTrain[vecinos[j]]]++;
	                  }              
	              }
	              priors[i][posClass] = ks[posClass]/ k;
	              priors[i][posClass ^ 1] = ks[posClass ^ 1]/ k;
	            }
	        }   
	    } else {
	        for (i=0; i<datosTrain.length; i++) {    
	                priors[i][clasesTrain[i]] = 1.0;
	                priors[i][clasesTrain[i] ^ 1] = 0.0;
	        } 
	    }
	    context.progress();
	    
	    
	    vecinos = new int[k];
	    prob = new double[nClases];
	    
	    // Index lists with elements marked for removal
	    ArrayList<Referencia> remove_positive = new ArrayList<Referencia>();
	    ArrayList<Referencia> remove_negative = new ArrayList<Referencia>();  

	    /*
	     * Body of the algorithm. 
	     * For each instance in T, search the correspond class conform his mayority 
	     * from the nearest neighborhood. 
	     * Is it is positive, the instance is selected.
	     */
	    for (i=0; i<datosTrain.length; i++) {    	
	      KNN.evaluacionKNN2(k, datosTrain, realTrain, nominalTrain, nulosTrain, 
	              clasesTrain, datosTrain[i], realTrain[i], nominalTrain[i], 
	              nulosTrain[i], nClases, distanceEu, vecinos);
	      
	      Arrays.fill(prob, 0.0);
	      for (j=0; j<vecinos.length; j++){
	    	  if (vecinos[j]>=0) {
	              prob[posClass] += priors[vecinos[j]][posClass] 
	                      / (1.0 + KNN.distancia(datosTrain[i], realTrain[i], 
	                      nominalTrain[i], nulosTrain[i], datosTrain[vecinos[j]], 
	                      realTrain[vecinos[j]], nominalTrain[vecinos[j]], 
	                      nulosTrain[vecinos[j]], distanceEu));
	              prob[posClass ^ 1] += priors[vecinos[j]][posClass ^ 1] 
	                      / (1.0 + KNN.distancia(datosTrain[i], realTrain[i], 
	                      nominalTrain[i], nulosTrain[i], datosTrain[vecinos[j]], 
	                      realTrain[vecinos[j]], nominalTrain[vecinos[j]], 
	                      nulosTrain[vecinos[j]], distanceEu));
	    	  }
	      }
	      sumProb = 0.0;
	      for (j=0; j<prob.length; j++) {
	    	  sumProb += prob[j];
	      }
	      for (j=0; j<prob.length; j++) {
	    	  prob[j] /= sumProb;
	      }
	      
	      maxProb = prob[0];
	      pos = 0;
	      for (j=1; j<prob.length; j++) {
	    	  if (prob[j] > maxProb) {
	              maxProb = prob[j];
	              pos = j;
	    	  }
	      }
	      
	      context.progress();
	      
	      claseObt = pos;
	      
	      if(version == 3 || version == 6){
	        if(claseObt != clasesTrain[i]){
	            if(clasesTrain[i] == posClass){
	                remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	            } else {
	                remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	            }
	        } else if (clasesTrain[i] != posClass 
	                && prob[posClass] > mu * prob[posClass ^ 1]){
	            remove_negative.add(new Referencia(i, 
	                        prob[posClass] - mu * prob[posClass ^ 1]));
	        }
	      } else if((version == 1 || version == 4) && claseObt != clasesTrain[i]){
	          if(clasesTrain[i] == posClass){
	             remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	          } else {
	              remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	          }
	      } else if((version == 2 || version == 5) 
	                            && (claseObt != clasesTrain[i] || maxProb <= mu)){
	          if(clasesTrain[i] == posClass){
	             remove_positive.add(new Referencia(i, 
	                        prob[posClass ^ 1] - prob[posClass]));
	          } else {
	              remove_negative.add(new Referencia(i, 
	                        prob[posClass] - prob[posClass ^ 1]));
	          }
	      }
	    }
	    
	    
	    /*
	     * Determine IR of S.
	     */
	    int nNeg = classDistr[posClass ^ 1] - remove_negative.size();
	    int nPos = classDistr[posClass] - remove_positive.size();
	    
	      System.out.println("nPos = " + nPos + ", nNeg = " + nNeg);
	    
	    if(nPos == 0 && nNeg == 0){ // everything removed
	            
	        // Select one random element of each class
	        do {
	            pos = Randomize.Randint(0,datosTrain.length-1);
	        } while (clasesTrain[pos] != posClass);

	        int neg;
	        do {
	            neg = Randomize.Randint(0,datosTrain.length-1);
	        } while (clasesTrain[neg] == posClass);

	        for(i = 0; i < marcas.length; i++){
	            marcas[i] = (i == pos || i == neg);
	        }
	        nSel = 2;

	    } else if(nPos > nNeg){ // Marks of negative instances need to be ignored

	         // Remove the appropriate positive instances
	        for(int s = 0; s < remove_positive.size(); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }

	        // Sort negative elements according to decreasing removal scores
	        Collections.sort(remove_negative); 

	        // Remove marked negative instances, ignoring the final part
	        for(int s = 0; s < classDistr[posClass ^ 1] - nPos; s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	    } else if(nPos == 0 || ((double) nNeg / nPos) > origIR){

	         // Remove the appropriate negative instances
	        for(int s = 0; s < remove_negative.size(); s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	        // Sort postive elements according to decreasing removal scores
	        Collections.sort(remove_positive); 

	        // Some marks of positive instances need to be ignored
	        int nRetain = (int) Math.ceil(nNeg / origIR);
	        for(int s = 0; s < Math.min(classDistr[posClass] - nRetain, 
	                remove_positive.size()); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }            
	    } else { // No problem, all marked instances can be removed.

	        // Positive instances
	        for(int s = 0; s < remove_positive.size(); s++){
	            marcas[remove_positive.get(s).entero] = false;
	            nSel--;
	        }

	        // Negative instances
	        for(int s = 0; s < remove_negative.size(); s++){
	            marcas[remove_negative.get(s).entero] = false;
	            nSel--;
	        }

	    } 
	    
	    context.progress();

	    /*Building of the S set from the flags*/
	    conjS = new double[nSel][datosTrain[0].length];
	    conjR = new double[nSel][datosTrain[0].length];
	    conjN = new int[nSel][datosTrain[0].length];
	    conjM = new boolean[nSel][datosTrain[0].length];
	    clasesS = new int[nSel];
	    for (i=0, l=0; i<datosTrain.length; i++) {
	      if (marcas[i]) { //the instance will be copied to the solution
	        for (j=0; j<datosTrain[0].length; j++) {
	          conjS[l][j] = datosTrain[i][j];
	          conjR[l][j] = realTrain[i][j];
	          conjN[l][j] = nominalTrain[i][j];
	          conjM[l][j] = nulosTrain[i][j];
	        }
	        clasesS[l] = clasesTrain[i];
	        l++;
	      }
	    }

	    System.out.println("ENNTh_Imb "+ relation + " " 
	            + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

	    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS, 
	            entradas, salida, nEntradas, relation);
	    OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, 
	            nEntradas, relation);
	  }

	  @SuppressWarnings("empty-statement")
	  @Override
	  public void leerConfiguracion (String ficheroScript) {

	    String fichero, linea, token;
	    StringTokenizer lineasFichero, tokens;
	    byte line[];
	    int i, j;

	    ficheroSalida = new String[2];

	    if(ficheroScript.equals("NOFILE")){
	    	System.out.println("There is no configuration file: Applying Auto-parameters");
	    	
	    	ficheroSalida[0] = "salida.dat";
	    	ficheroSalida[1] = "otro.dat";
	    	ficheroTraining = "intermediate.dat";	    	
	    	
	    	this.k = 3;
	    	this.distanceEu = true;
			this.mu = 0.7;
			this.version = 5;			
	    	
	    } else {
		    fichero = Fichero.leeFichero (ficheroScript);
		    lineasFichero = new StringTokenizer (fichero,"\n\r");
	
		    lineasFichero.nextToken();
		    linea = lineasFichero.nextToken();
	
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    token = tokens.nextToken();
	
		    /*Getting the names of the training and test files*/
		    line = token.getBytes();
		    for (i=0; line[i]!='\"'; i++);
		    i++;
		    for (j=i; line[j]!='\"'; j++);
		    ficheroTraining = new String (line,i,j-i);
		    for (i=j+1; line[i]!='\"'; i++);
		    i++;
		    for (j=i; line[j]!='\"'; j++);
		    ficheroTest = new String (line,i,j-i);
	
		    /*Getting the path and base name of the results files*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    token = tokens.nextToken();
	
		    /*Getting the names of output files*/
		    line = token.getBytes();
		    for (i=0; line[i]!='\"'; i++);
		    i++;
		    for (j=i; line[j]!='\"'; j++);
		    ficheroSalida[0] = new String (line,i,j-i);
		    for (i=j+1; line[i]!='\"'; i++);
		    i++;
		    for (j=i; line[j]!='\"'; j++);
		    ficheroSalida[1] = new String (line,i,j-i);
		    
		    /*Getting the number of neighbors*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    k = Integer.parseInt(tokens.nextToken().substring(1));
	
		    /*Getting the noise threshold*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    mu = Double.parseDouble(tokens.nextToken().substring(1));
	
		    /*Getting the type of distance function*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;  
		    
		    /*Which version to perform*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    version = Integer.parseInt(tokens.nextToken().substring(1));
	    }
	}

}
