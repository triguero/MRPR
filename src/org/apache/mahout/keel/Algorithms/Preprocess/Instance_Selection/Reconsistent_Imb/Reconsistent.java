package org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.Reconsistent_Imb;

import java.util.ArrayList;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.*;

import java.util.StringTokenizer;
import java.util.Vector;
import java.util.Arrays;
import java.util.Collections;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.*;

public class Reconsistent extends Metodo {
	
	/*Own parameters of the algorithm*/
	  private int k;

	  public Reconsistent (String ficheroScript) {
	    super (ficheroScript);
	  }
	  
	  public Reconsistent (String ficheroScript, InstanceSet train) {
		 super (ficheroScript, train);
	  }

	  public void ejecutar (Context context) {
		  
		  System.out.println("I am executing Reconsistent_Imb");

	    boolean marcas[];
	    boolean marcastmp[];
	    boolean set_M[];
	    int nSel;
	    double conjS[][];
	    double conjR[][];
	    int conjN[][];
	    boolean conjM[][];
	    int clasesS[];
	    Vector <Integer> vecinos[];
	    int next;
	    int maxneigh;
	    int claseObt;
	    int nClases;

	    long tiempo = System.currentTimeMillis();
	    
	    /*Getting the number of differents classes*/
	    nClases = 0;
	    for (int i=0; i<clasesTrain.length; i++)
	      if (clasesTrain[i] > nClases)
	        nClases = clasesTrain[i];
	    nClases++;    
	    
	    /*
	     * Determine minority and majority class (binary) and IR 
	     * of the original set.
	     */
	    int[] classDistr = new int[nClases];
	    for (int i = 0; i < clasesTrain.length; i++){
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
	    
	    /*
	     * PHASE 1
	     */
	    marcas = new boolean[datosTrain.length];
	    set_M = new boolean[datosTrain.length];
	    marcastmp = new boolean[datosTrain.length];
	    Arrays.fill(marcas, true);
	    Arrays.fill(set_M, false);
	    Arrays.fill(marcastmp, true);
	    vecinos = new Vector [datosTrain.length];
	    for (int i=0; i<datosTrain.length; i++)
	        vecinos[i] = new Vector <Integer>();

	    // Find the neighborhoods
	    for (int i=0; i<datosTrain.length; i++) {
	        next = nextNeighbour (marcas, datosTrain, realTrain, nominalTrain, 
	                nulosTrain, i, vecinos[i]);
	        for (int j=0; j<datosTrain.length; j++)
	            marcastmp[j] = marcas[j];
	        while (next >= 0 && clasesTrain[next] == clasesTrain[i]) {
	        	vecinos[i].add(new Integer(next));
	        	marcastmp[next] = false;
	                next = nextNeighbour(marcastmp,datosTrain, realTrain, 
	                        nominalTrain, nulosTrain,i,vecinos[i]);
	        }    
	        context.progress();
	    }
	    
	    
	    
	    // Flags protecting elements from removal
	    boolean[] always_select_this_element = new boolean[datosTrain.length];
	    
	    // Determine initial neighborhood sizes
	    int[] n_sizes = new int[datosTrain.length];
	    for(int i=0; i < datosTrain.length; i++){
	        n_sizes[i] = vecinos[i].size();
	        if(n_sizes[i] == 0){
	            always_select_this_element[i] = true;
	        }
	    }

	    // R-values 
	    int[] r_values = new int[datosTrain.length];
	    
	    // First candidate
	    int x = 0;   
	    boolean found_cand = true;
	    maxneigh = n_sizes[0];
	    for (int i=1; i<datosTrain.length; i++) {
	      if (n_sizes[i] > maxneigh) {
	        maxneigh = vecinos[i].size();
	        x = i;
	      }
	    }
	    
	    while (found_cand) {
	    	context.progress();
	        always_select_this_element[x] = true;
	        
	        for (int i = 0; i < vecinos[x].size(); i++) {
	            
	            // Get the neighbor
	            int neighbor = vecinos[x].elementAt(i).intValue();
	            
	            // Increase its r-value
	            r_values[neighbor]++;
	            
	            // Decrease size of the neighborhoods to which our neighbor belongs
	            for (int z = 0; z < datosTrain.length; z++) {
	                if(vecinos[z].contains((Integer) neighbor)){
	                    n_sizes[z]--;
	                }
	            }
	        }

	        // Find the next candidate
	        x = next_candidate(n_sizes, r_values, always_select_this_element);
	        found_cand = (x >= 0);      
	    }
	    
	    // Construction of S
	    Arrays.fill(marcas, false);
	    nSel = 0;
	    int nPos = 0;
	    int nNeg = 0;
	    for(int i = 0; i < datosTrain.length; i++){
	        marcas[i] = (always_select_this_element[i] || r_values[i] == 0);
	        nSel++;
	        if(marcas[i]){
	            if(clasesTrain[i] == posClass){
	                nPos++;
	            } else {
	                nNeg++;
	            }
	        }
	    }
	    context.progress();
	    
	    // Assess class distribution in S
	    if(nPos > nNeg){
	        // Add more negative elements
	        ArrayList<Referencia> neg_els = new ArrayList<Referencia>();
	        
	        for(int i = 0; i < datosTrain.length; i++){
	            if(!marcas[i] && clasesTrain[i] != posClass){
	                neg_els.add(new Referencia(i, r_values[i]));
	            }
	        }
	        
	        // Sort in increasing order of r_values
	        Collections.sort(neg_els, Collections.reverseOrder());
	        
	        int c = 0;
	        while(c < neg_els.size() && nNeg < nPos){
	            marcas[neg_els.get(c).entero] = true;
	            nNeg++;
	            nSel++;
	            c++;
	        }
	                
	        
	    } else if((double) nNeg / nPos > origIR){
	        // Add more positive elements
	        ArrayList<Referencia> pos_els = new ArrayList<Referencia>();
	        
	        for(int i = 0; i < datosTrain.length; i++){
	            if(!marcas[i] && clasesTrain[i] == posClass){
	                pos_els.add(new Referencia(i, r_values[i]));
	            }
	        }
	        
	        // Sort in increasing order of r_values
	        Collections.sort(pos_els, Collections.reverseOrder());
	        
	        int c = 0;
	        while(c < pos_els.size() && (double) nNeg / nPos > origIR){
	            marcas[pos_els.get(c).entero] = true;
	            nPos++;
	            nSel++;
	            c++;
	        }        
	    }    
	    
	    conjS = new double[nSel][datosTrain[0].length];
	    conjR = new double[nSel][datosTrain[0].length];
	    conjN = new int[nSel][datosTrain[0].length];
	    conjM = new boolean[nSel][datosTrain[0].length];
	    clasesS = new int[nSel];
	    for (int i=0, l=0; i<datosTrain.length; i++) {
	      if (marcas[i]) { //the instance will be copied to the solution
	        for (int j=0; j<datosTrain[0].length; j++) {
	          conjS[l][j] = datosTrain[i][j];
	          conjR[l][j] = realTrain[i][j];
	          conjN[l][j] = nominalTrain[i][j];
	          conjM[l][j] = nulosTrain[i][j];
	        }
	        clasesS[l] = clasesTrain[i];
	        l++;
	      }
	    }
	    context.progress();

	    /*
	     * PHASE 2
	     */
	    
	    // Determine the set M
	    for (int i = 0; i < datosTrain.length; i++) {
	        // Classify with kNN
	        claseObt = KNN.evaluacionKNN2 (k, conjS, conjR, conjN, conjM, 
	                clasesS, datosTrain[i], realTrain[i], nominalTrain[i],
	                nulosTrain[i], nClases, distanceEu);
	        
	        // Add to M if misclassified
	        set_M[i] = (claseObt != clasesTrain[i]);
	        context.progress();
	    }

	    // Construct neighborhoods for instances in M
	    Vector <Integer> vecinos_own[] = new Vector [datosTrain.length];
	    Vector <Integer> vecinos_other[] = new Vector [datosTrain.length];
	    for (int i=0; i<datosTrain.length; i++){
	        vecinos_own[i] = new Vector <Integer>();
	        vecinos_other[i] = new Vector <Integer>();
	    }
	    
	    int[] n_own = new int[datosTrain.length];
	    int[] n_other = new int[datosTrain.length];
	    
	    ArrayList<Referencia> order = new ArrayList<Referencia>();
	    
	    for (int i=0; i<datosTrain.length; i++) {
	    	context.progress();
	        if(set_M[i]){
	            
	            // Neighborhood of own class
	            int own_class = clasesTrain[i];
	            next = nextNeighbour_phase2(i, vecinos_own[i], own_class);
	            while (next >= 0 && clasesTrain[next] == own_class) {
	                vecinos_own[i].add(new Integer(next));
	                n_own[next]++;
	                next = nextNeighbour_phase2(i, vecinos_own[i], own_class);
	            } 
	            
	            order.add(new Referencia(i, vecinos_own[i].size()));
	            
	            // Neighborhood of opposite class
	            int other_class = own_class ^ 1;
	            next = nextNeighbour_phase2(i, vecinos_other[i], other_class);
	            while (next >= 0 && clasesTrain[next] == other_class) {
	                vecinos_other[i].add(new Integer(next));
	                n_other[next]++;                   
	                next = nextNeighbour_phase2(i, vecinos_other[i], other_class);
	            } 
	        }
	         
	    }
	    
	    boolean[] marcas_phase2 = new boolean[datosTrain.length];
	    
	    // Sort the instances of M
	    Collections.sort(order);
	    context.progress();
	    
	    // Select at most one instance for each element of M
	    for(int i = 0; i < order.size(); i++){
	        int this_el = order.get(i).entero;
	        int to_add = select_element_phase2(marcas, marcas_phase2, 
	                    vecinos_own[this_el], n_own, n_other, this_el);
	        if(to_add >= 0){
	            marcas_phase2[to_add] = true;
	        }
	    }
	    
	    context.progress();
	    
	    for (int i=0; i<marcas.length; i++){
	    	marcas[i] = marcas[i] || marcas_phase2[i];
	    }
	    

	    /*Building of the S set from the flags*/
	    nSel = 0;
	    for (int i=0; i<datosTrain.length; i++)
	      if (marcas[i]) nSel++;
	    conjS = new double[nSel][datosTrain[0].length];
	    conjR = new double[nSel][datosTrain[0].length];
	    conjN = new int[nSel][datosTrain[0].length];
	    conjM = new boolean[nSel][datosTrain[0].length];
	    clasesS = new int[nSel];
	    for (int i=0, l=0; i<datosTrain.length; i++) {
	      if (marcas[i]) { //the instance will be copied to the solution
	        for (int j=0; j<datosTrain[0].length; j++) {
	          conjS[l][j] = datosTrain[i][j];
	          conjR[l][j] = realTrain[i][j];
	          conjN[l][j] = nominalTrain[i][j];
	          conjM[l][j] = nulosTrain[i][j];
	        }
	        clasesS[l] = clasesTrain[i];
	        l++;
	      }
	    }
	    context.progress();

	    System.out.println("Reconsistent_Imb "+ relation + 
	            " " + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

	    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, 
	            clasesS, entradas, salida, nEntradas, relation);
	    OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, 
	            nEntradas, relation);
	  }

	    // SARAH
	    int nextNeighbour (boolean marcas[], double datos[][], 
	            double datosR[][], int datosN[][], boolean datosM[][], 
	            int ej, Vector <Integer> vecinos) {

		  int i, j, k;
		  int pos = -1;
		  double distmin = Double.POSITIVE_INFINITY;
		  double distancia;
		  double centroid[];
	          double centroidR[];    // SARAH
	          int centroidN[];       // SARAH
	          boolean centroidM[];   // SARAH
		  double prototipo[];
	          double prototipoR[];   // SARAH
	          int prototipoN[];      // SARAH
	          boolean prototipoM[];  // SARAH

		  /*Computation of the previous centroid*/
		  centroid = new double[datos[0].length];
	          centroidR = new double[datos[0].length];
	          centroidN = new int[datos[0].length];
	          centroidM = new boolean[datos[0].length];
		  prototipo = new double[datos[0].length];
	          prototipoR = new double[datos[0].length];
	          prototipoN = new int[datos[0].length];
	          prototipoM = new boolean[datos[0].length];
	          
	          Arrays.fill(centroid, 0.0);
	          Arrays.fill(centroidR, 0.0);
	          Arrays.fill(centroidN, 0);
	          Arrays.fill(centroidM, false);
	          
	          int[][] votesForNominal = new int[datos[0].length][];
		  
	          // First consider the nominal attributes
	          for (j=0; j<datos[0].length; j++) {
	                    if(Attributes.getInputAttribute(j).getType() == Attribute.NOMINAL){
	                        votesForNominal[j] = new int[Attributes.getInputAttribute(j).getNominalValuesList().size()];
	                        int winner = 0; // position containing the maximum number of votes
	                        int votesWinner = 0;
	                        for (k=0; k<vecinos.size(); k++) {
	                                if(datosM[vecinos.elementAt(k).intValue()][j]){
	                                    centroidM[j] = true;
	                                 }
	                                int nomValue = datosN[vecinos.elementAt(k).intValue()][j];
	                                votesForNominal[j][nomValue]++;
	                                if(votesForNominal[j][nomValue] > votesWinner){
	                                    winner = nomValue;
	                                    votesWinner = votesForNominal[j][nomValue];
	                                }                                    
	                        }                        
	                        centroidN[j] = winner;
	                    }
	            }
	          
	            // Continue with the non-nominal attributes  
	            for (k=0; k<vecinos.size(); k++) {
	                    for (j=0; j<datos[0].length; j++) {
	                        if(Attributes.getInputAttribute(j).getType() != Attribute.NOMINAL){
	                                if(datosM[vecinos.elementAt(k).intValue()][j]){
	                                    centroidM[j] = true;
	                                }
	                                centroidR[j] += datosR[vecinos.elementAt(k).intValue()][j];
	                            }
	                    }
	            }           
	      	  
	          // Locate the new neighbor
		  for (i=0; i<datos.length; i++) {              
	              System.arraycopy(centroidM, 0, prototipoM, 0, centroidM.length);
		      if (marcas[i] && i != ej) {
		    	  for (j=0; j<datos[0].length; j++) {
	                      if(datosM[i][j]){
	                         prototipoM[j] = true;
	                      }
	                      if(Attributes.getInputAttribute(j).getType() == Attribute.NOMINAL){
	                         int additional = datosN[i][j];
	                         votesForNominal[j][additional]++;
	                         if(votesForNominal[j][additional] > votesForNominal[j][centroidN[j]]){
	                             prototipoN[j] = additional;
	                         } else {
	                             prototipoN[j] = centroidN[j];
	                         }
	                         if(Attributes.getInputAttribute(j).
	                            getNominalValuesList().size() == 1){
	                             prototipo[j] = (double) prototipoN[j];
	                         } else {
	                             prototipo[j] = (double) prototipoN[j] / (Attributes.getInputAttribute(j).
	                            getNominalValuesList().size() - 1);
	                         }
	                         
	                      } else {
	                        prototipo[j] = centroid[j] + datosR[i][j];
	                        
	                        prototipo[j] /= (vecinos.size()+1);
	                                        
	                        //  Normalization
	                        prototipo[j] = 
	                                prototipoR[j] - Attributes.getInputAttribute(j).getMinAttribute();
	                        prototipo[j] /= 
	                                Attributes.getInputAttribute(j).getMaxAttribute() - Attributes.getInputAttribute(j).getMinAttribute();
	                        if (Double.isNaN(prototipo[j])){
	                            prototipo[j] = prototipoR[j];
	                        }
	                      }
		    	  }  
	                  
		          distancia = KNN.distancia (datos[ej], datosR[ej], datosN[ej], 
	                          datosM[ej], prototipo, prototipoR, prototipoN, 
	                          prototipoM, distanceEu);
		          if (distancia < distmin) {
		              distmin = distancia;
		              pos = i;
		          }
		      }
		  }
		    
		  return pos;

	  }
	    
	  private int nextNeighbour_phase2 (int ej, Vector <Integer> vecinos, 
	          int this_class) {

		  int i, j, k;
		  int pos = -1;
		  double distmin = Double.POSITIVE_INFINITY;
		  double distancia;
		  double centroid[];
	          double centroidR[];    // SARAH
	          int centroidN[];       // SARAH
	          boolean centroidM[];   // SARAH
		  double prototipo[];
	          double prototipoR[];   // SARAH
	          int prototipoN[];      // SARAH
	          boolean prototipoM[];  // SARAH
	          
	          if(vecinos.isEmpty()){
	              /*
	               * This is the first neighbor to be added.
	               * Select the nearest neighhbor of the requested class
	               */
	               double minDist = Double.POSITIVE_INFINITY;
	               for (j=0; j<datosTrain.length; j++) {
	                    if(j != ej && clasesTrain[j] == this_class){
	                        distancia = KNN.distancia(datosTrain[ej], realTrain[ej], 
	                              nominalTrain[ej], nulosTrain[ej], datosTrain[j], 
	                              realTrain[j], nominalTrain[j], nulosTrain[j], 
	                              distanceEu);
	                        if(distancia < minDist){
	                            minDist = distancia;
	                            pos = j;
	                        }
	                    }
	              }

	          } else {

	              /*Computation of the previous centroid*/
	              centroid = new double[datosTrain[0].length];
	              centroidR = new double[datosTrain[0].length];
	              centroidN = new int[datosTrain[0].length];
	              centroidM = new boolean[datosTrain[0].length];
	              prototipo = new double[datosTrain[0].length];
	              prototipoR = new double[datosTrain[0].length];
	              prototipoN = new int[datosTrain[0].length];
	              prototipoM = new boolean[datosTrain[0].length];

	              Arrays.fill(centroid, 0.0);
	              Arrays.fill(centroidR, 0.0);
	              Arrays.fill(centroidN, 0);
	              Arrays.fill(centroidM, false);

	              int[][] votesForNominal = new int[datosTrain[0].length][];

	              // First consider the nominal attributes
	              for (j=0; j<datosTrain[0].length; j++) {
	                    if(Attributes.getInputAttribute(j).getType() == Attribute.NOMINAL){
	                        votesForNominal[j] = new int[Attributes.getInputAttribute(j).getNominalValuesList().size()];
	                        int winner = 0; // position containing the maximum number of votes
	                        int votesWinner = 0;
	                        for (k=0; k<vecinos.size(); k++) {
	                                if(nulosTrain[vecinos.elementAt(k).intValue()][j]){
	                                    centroidM[j] = true;
	                                 }
	                                int nomValue = nominalTrain[vecinos.elementAt(k).intValue()][j];
	                                votesForNominal[j][nomValue]++;
	                                if(votesForNominal[j][nomValue] > votesWinner){
	                                    winner = nomValue;
	                                    votesWinner = votesForNominal[j][nomValue];
	                                }                                    
	                        }                        
	                        centroidN[j] = winner;
	                    }
	                }

	                // Continue with the non-nominal attributes  
	                for (k=0; k<vecinos.size(); k++) {
	                    for (j=0; j<datosTrain[0].length; j++) {
	                        if(Attributes.getInputAttribute(j).getType() != Attribute.NOMINAL){
	                                if(nulosTrain[vecinos.elementAt(k).intValue()][j]){
	                                    centroidM[j] = true;
	                                }
	                                centroidR[j] += realTrain[vecinos.elementAt(k).intValue()][j];
	                        }
	                    }
	                }           

	              // Locate the new neighbor
	              for (i=0; i<datosTrain.length; i++) {              
	                  System.arraycopy(centroidM, 0, prototipoM, 0, centroidM.length);
	                  if (i != ej && !vecinos.contains((Integer) i)) {
	                      for (j=0; j<datosTrain[0].length; j++) {
	                          if(nulosTrain[i][j]){
	                             prototipoM[j] = true;
	                          }
	                          if(Attributes.getInputAttribute(j).getType() == Attribute.NOMINAL){
	                             int additional = nominalTrain[i][j];
	                             votesForNominal[j][additional]++;
	                             if(votesForNominal[j][additional] > votesForNominal[j][centroidN[j]]){
	                                 prototipoN[j] = additional;
	                             } else {
	                                 prototipoN[j] = centroidN[j];
	                             }
	                             if(Attributes.getInputAttribute(j).
	                                getNominalValuesList().size() == 1){
	                                 prototipo[j] = (double) prototipoN[j];
	                             } else {
	                                 prototipo[j] = (double) prototipoN[j] / (Attributes.getInputAttribute(j).
	                                getNominalValuesList().size() - 1);
	                             }

	                          } else {
	                            prototipo[j] = centroid[j] + realTrain[i][j];

	                            prototipo[j] /= (vecinos.size()+1);

	                            //  Normalization
	                            prototipo[j] = 
	                                    prototipoR[j] - Attributes.getInputAttribute(j).getMinAttribute();
	                            prototipo[j] /= 
	                                    Attributes.getInputAttribute(j).getMaxAttribute() - Attributes.getInputAttribute(j).getMinAttribute();
	                            if (Double.isNaN(prototipo[j])){
	                                prototipo[j] = prototipoR[j];
	                            }
	                          }
	                      }  

	                      distancia = KNN.distancia (datosTrain[ej], realTrain[ej], 
	                              nominalTrain[ej], nulosTrain[ej], prototipo, 
	                              prototipoR, prototipoN, prototipoM, distanceEu);
	                      if (distancia < distmin) {
	                          distmin = distancia;
	                          pos = i;
	                      }
	                  }
	              }
	          }
		    
		  return pos;

	  }


	  @Override
	  @SuppressWarnings("empty-statement")
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
	    	
	    	this.k = 1;
	    	this.distanceEu = true;		
	    	
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
		    
		    /*Getting the type of distance function*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;  
	    }
	    
	  }
	  
	  private boolean notANeighbor(int cand, int[] nvecinos) {
	        boolean not_neighbor = true;
	        int c = 0;
	        while(c < nvecinos.length && not_neighbor){
	            not_neighbor = (cand != nvecinos[c]);
	            c++;
	        }
	        return not_neighbor;
	    }

	    private int next_candidate(int[] n_sizes, int[] r_values, 
	            boolean[] always_select_this_element) {
	        int cand = -1;
	        
	        int size = 0;
	        for (int i = 0; i < n_sizes.length; i++){
	            if(!always_select_this_element[i] && r_values[i] == 0 &&
	                    n_sizes[i] > size){
	                cand = i;
	                size = n_sizes[i];
	            }
	        }

	        return cand;
	    }

	    private int select_element_phase2(boolean[] already_in_S, 
	            boolean[] selected_phase2, Vector<Integer> candidates, 
	            int[] n_own, int[] n_other, int instance) {
	       int to_add = -1;
	       
	       // Sort the neighbors
	       ArrayList<Referencia> order = new ArrayList<Referencia>();
	       for(int i = 0; i < candidates.size(); i++){
	           order.add(new Referencia(candidates.get(i), n_own[i] - n_other[i]));
	       }
	       
	       Collections.sort(order);
	       
	       int c = 0;
	       boolean stop = false;
	       while(c < order.size() && !stop){
	           int cand = order.get(c).entero;
	           if(!already_in_S[cand] && !selected_phase2[cand]){
	               to_add = cand;
	               stop = true;
	           } else if(selected_phase2[cand]){
	               stop = true;
	           } else {
	              c++; 
	           }           
	       }
	       
	       if(c == order.size() && !already_in_S[instance] 
	               && !selected_phase2[instance]){ // Instance itself will be added
	           to_add = instance;
	       }
	       
	       return to_add;
	    }

}
