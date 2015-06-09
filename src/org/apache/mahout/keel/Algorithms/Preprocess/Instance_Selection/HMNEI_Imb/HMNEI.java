package org.apache.mahout.keel.Algorithms.Preprocess.Instance_Selection.HMNEI_Imb;

import java.util.Arrays;

import org.apache.mahout.keel.Algorithms.Preprocess.Basic.*;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.core.*;
import org.apache.hadoop.mapreduce.Mapper.Context;

import java.util.StringTokenizer;

public class HMNEI extends Metodo {

	
	/*Own parameters of the algorithm*/
	  private double epsilon;
	  private int version;
	  
	  public HMNEI (String ficheroScript) {
	    super (ficheroScript);
	  }
	  
	  public HMNEI (String ficheroScript, InstanceSet train) {
	    super (ficheroScript, train);
      }

	  public void ejecutar (Context context) {
		  
		 System.out.println("I am executing HMNEI_Imb");

	    int i, j, k, l, m;
	    int nClases;
	    int claseObt;
	    boolean marcas[];
	    int nSel = 0;
	    double conjS[][];
	    double conjR[][];
	    int conjN[][];
	    boolean conjM[][];
	    int clasesS[];
	    double conjS2[][];
	    double conjR2[][];
	    int conjN2[][];
	    boolean conjM2[][];
	    int clasesS2[];
	    double dist, minDist;
	    double gmean, gmeanAct = 0.0;
	    double acierto, aciertoAct = 0.0;
	    int hit[], miss[];
	    int pos, cont;
	    double w[];
	    int cc[];
	    int seleccionadosAnt;
	    
	    boolean stop = false;

	    long tiempo = System.currentTimeMillis();

	    /*Getting the number of differents classes*/
	    nClases = 0;
	    for (i=0; i<clasesTrain.length; i++)
	      if (clasesTrain[i] > nClases)
	        nClases = clasesTrain[i];
	    nClases++;
	    
	    
	    /*Building of the S set from the flags*/
	    conjS2 = new double[datosTrain.length][datosTrain[0].length];
	    conjR2 = new double[datosTrain.length][datosTrain[0].length];
	    conjN2 = new int[datosTrain.length][datosTrain[0].length];
	    conjM2 = new boolean[datosTrain.length][datosTrain[0].length];
	    clasesS2 = new int[datosTrain.length];
	    for (m=0, l=0; m<datosTrain.length; m++) {
	            for (j=0; j<datosTrain[0].length; j++) {
	                    conjS2[l][j] = datosTrain[m][j];
	                    conjR2[l][j] = realTrain[m][j];
	                    conjN2[l][j] = nominalTrain[m][j];
	                    conjM2[l][j] = nulosTrain[m][j];
	            }
	            clasesS2[l] = clasesTrain[m];
	            l++;
	    }    

	    nSel = datosTrain.length;
	    
	    context.progress();
	    
	    Referencia[] scores; // will store the indegrees
	    
	    /*
	     * Determine minority and majority class (binary) and IR 
	     * of the previous set.
	     */
	    int[] orig_distr = new int[nClases];
	    for (i = 0; i < clasesTrain.length; i++){
	        orig_distr[clasesTrain[i]]++;
	    }
	    int orig_posClass;
	    if(orig_distr[0] < orig_distr[1]){
	        orig_posClass = 0;
	    } else {
	        orig_posClass = 1;
	    }
	    int nPos = orig_distr[orig_posClass];
	    int nNeg = orig_distr[orig_posClass ^ 1];
	    

	    if(version == 1){
	        do {
	            gmean = gmeanAct;
	            seleccionadosAnt = nSel;

	            /* Copy of the previous set S. */
	            conjS = new double[nSel][datosTrain[0].length];
	            conjR = new double[nSel][datosTrain[0].length];
	            conjN = new int[nSel][datosTrain[0].length];
	            conjM = new boolean[nSel][datosTrain[0].length];
	            clasesS = new int[nSel];
	            for (m=0, l=0; m<nSel; m++) {
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS[l][j] = conjS2[m][j];
	                            conjR[l][j] = conjR2[m][j];
	                            conjN[l][j] = conjN2[m][j];
	                            conjM[l][j] = conjM2[m][j];
	                    }
	                    clasesS[l] = clasesS2[m];
	                    l++;
	            }
	            
	            context.progress();

	            /*
	             * Determine minority and majority class (binary) and IR 
	             * of the previous set.
	             */
	            int[] classDistr = new int[nClases];
	            for (i = 0; i < nSel; i++){
	                classDistr[clasesS[i]]++;
	            }
	            int posClass;
	            if(classDistr[0] < classDistr[1]){
	                posClass = 0;
	            } else {
	                posClass = 1;
	            }
	            double origIR = (double) classDistr[posClass ^ 1]/ classDistr[posClass];

	            /*Inicialization of the flagged instances vector from the S*/
	            marcas = new boolean[nSel];
	            for (i=0; i<nSel; i++) {
	                    marcas[i] = true;
	            }

	            hit = new int[nSel];
	            miss = new int[nSel];
	            for (i=0; i<conjS.length; i++) {
	                for (j=0; j<nClases; j++) {
	                    minDist = Double.POSITIVE_INFINITY;
	                    pos = -1;
	                    for (k=0; k<conjS.length; k++) {
	                        if (i!=k && clasesS[k] == j) {
	                            dist = KNN.distancia(conjS[i], conjR[i], 
	                                    conjN[i], conjM[i], conjS[k], conjR[k], 
	                                    conjN[k], conjM[k], distanceEu);
	                            if (dist < minDist) {
	                                minDist = dist;    						
	                                pos = k;
	                            }
	                        }
	                    }
	                    if (pos >= 0) {
	                        if (clasesS[i] == j) {
	                            hit[pos]++;
	                        } else {
	                            miss[pos]++;
	                        }
	                    }
	                }
	                
	                context.progress();
	            }

	            w = new double[nClases];
	            cc = new int[nClases];
	            for (i=0; i<w.length; i++) {
	                    cont = 0;
	                    for (j=0; j<clasesS.length; j++) {
	                            if (clasesS[j] == i) {
	                                    cont++;
	                            }
	                    }
	                    cc[i] = cont;
	                    w[i] = (double)cont / (double)nSel;
	            }

	            /*RULE H1*/        
	            for (i=0; i<hit.length; i++) {
	                if ((w[clasesS[i]] * (double)miss[i] + epsilon) > 
	                        ((1-w[clasesS[i]]) * (double)hit[i])) {
	                    marcas[i] = false;
	                    nSel--;
	                }
	            }
	            context.progress();

	            /*RULE H3*/
	            if (nClases > 3) {
	                for (i=0; i<hit.length; i++) {
	                    if (!marcas[i] && (miss[i]+hit[i] > 0) 
	                            && miss[i] < (nClases/2)) {
	                        marcas[i] = true;
	                        nSel++;
	                    }
	                }
	            }
	            context.progress();


	            /*RULE H4*/
	            for (i=0; i<hit.length; i++) {
	                if (!marcas[i] && hit[i] >= (cc[clasesS[i]] / 4)) {
	                    marcas[i] = true;
	                    nSel++;
	                }
	            }
	            context.progress();

	            /*
	             * RULE H2_Imb.
	             * Determine IR in the new S. 
	             * When it is higher than that of the previous set, this wlll be 
	             * resolved by adding additional elements in decreasing order of 
	             * their indegree.
	             * Positive class is also not allowed to become the absolute majority.
	             */
	            int nPosS = 0;
	            int nNegS = 0;
	            for(i = 0; i < hit.length; i++){
	                if(marcas[i]){
	                    if(clasesS[i] == posClass){
	                        nPosS++;
	                    } else {
	                        nNegS++;  
	                    }
	                }
	            }
	            context.progress();

	            if(nPosS == 0 && nNegS == 0){ // everything removed

	                // Select one random element of each class and stop after this iteration
	                int posi;
	                do {
	                    posi = Randomize.Randint(0,hit.length-1);
	                } while (clasesS[posi] != posClass);

	                int nega;
	                do {
	                    nega = Randomize.Randint(0,hit.length-1);
	                } while (clasesS[nega] == posClass);


	                marcas[posi] = true;
	                marcas[nega] = true;

	                nSel = 2;
	                stop = true;
	                
	                context.progress();

	            } else if(nPosS > nNegS){ 
	                // Some negative instances need to be reselected, until IR = 1.
	                scores = new Referencia[hit.length]; 

	                for(i=0; i < hit.length; i++){
	                    scores[i] = new Referencia(i, hit[i] + miss[i]);
	                }

	                Arrays.sort(scores);

	                // Undo marks of negative instances until nNegS == nPosS
	                int c = 0;
	                while(nNegS < nPosS && c < scores.length){
	                    if(clasesS[scores[c].entero] != posClass 
	                            && !marcas[scores[c].entero]){
	                        marcas[scores[c].entero] = true;
	                        nSel++;
	                        nNegS++;
	                    }
	                    c++;
	                }
	                
	                context.progress();


	            } else if(nPosS == 0 || ((double) nNegS / nPosS) > origIR){
	                // Some positive instances need to be reselected
	                scores = new Referencia[hit.length]; 

	                for(i=0; i < hit.length; i++){
	                    scores[i] = new Referencia(i, hit[i] + miss[i]);
	                }

	                Arrays.sort(scores);

	                // Undo marks of negative instances until nNegS == nPosS
	                int c = 0;
	                while(((double) nNegS / nPosS) > origIR && c < scores.length){
	                    if(clasesS[scores[c].entero] == posClass 
	                            && !marcas[scores[c].entero]){
	                        marcas[scores[c].entero] = true;
	                        nSel++;
	                        nPosS++;
	                    }
	                    c++;
	                }
	                
	                context.progress();
	            }  


	            /* The new prototype set */
	            conjS2 = new double[nSel][datosTrain[0].length];
	            conjR2 = new double[nSel][datosTrain[0].length];
	            conjN2 = new int[nSel][datosTrain[0].length];
	            conjM2 = new boolean[nSel][datosTrain[0].length];
	            clasesS2 = new int[nSel];
	            for (m=0, l=0; m<conjS.length; m++) {
	                if (marcas[m]) { //the instance will be evaluated
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS2[l][j] = conjS[m][j];
	                            conjR2[l][j] = conjR[m][j];
	                            conjN2[l][j] = conjN[m][j];
	                            conjM2[l][j] = conjM[m][j];
	                    }
	                    clasesS2[l] = clasesS[m];
	                    l++;
	                }
	            }
	            
	            context.progress();

	            // Determine gmean of the new set
	            int tn = 0;
	            int tp = 0;
	            for (i=0; i<datosTrain.length; i++) {
	                claseObt = KNN.evaluacionKNN2(1, conjS2, conjR2, conjN2, 
	                        conjM2, clasesS2, datosTrain[i], realTrain[i],
	                        nominalTrain[i], nulosTrain[i], nClases, distanceEu);
	                if (claseObt == clasesTrain[i]) {  // Classified correctly
	                    if(clasesTrain[i] == orig_posClass){ // true positive
	                        tp++;
	                    } else { // true negative
	                        tn++;
	                    }
	                }
	            }
	            double tpr = (double) tp / nPos;
	            double tnr = (double) tn / nNeg;
	            gmeanAct = Math.sqrt(tpr * tnr);
	        } while (gmeanAct >= gmean && nSel < seleccionadosAnt && !stop);
	    
	    } else if(version == 2){
	        do {
	            acierto = aciertoAct;
	            seleccionadosAnt = nSel;

	            /* Copy of the previous set S. */
	            conjS = new double[nSel][datosTrain[0].length];
	            conjR = new double[nSel][datosTrain[0].length];
	            conjN = new int[nSel][datosTrain[0].length];
	            conjM = new boolean[nSel][datosTrain[0].length];
	            clasesS = new int[nSel];
	            for (m=0, l=0; m<nSel; m++) {
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS[l][j] = conjS2[m][j];
	                            conjR[l][j] = conjR2[m][j];
	                            conjN[l][j] = conjN2[m][j];
	                            conjM[l][j] = conjM2[m][j];
	                    }
	                    clasesS[l] = clasesS2[m];
	                    l++;
	            }
	            
	            context.progress();

	            /*
	             * Determine minority and majority class (binary) and IR 
	             * of the previous set.
	             */
	            int[] classDistr = new int[nClases];
	            for (i = 0; i < nSel; i++){
	                classDistr[clasesS[i]]++;
	            }
	            int posClass;
	            if(classDistr[0] < classDistr[1]){
	                posClass = 0;
	            } else {
	                posClass = 1;
	            }
	            double origIR = (double) classDistr[posClass ^ 1]/ classDistr[posClass];

	            /*Inicialization of the flagged instances vector from the S*/
	            marcas = new boolean[nSel];
	            for (i=0; i<nSel; i++) {
	                    marcas[i] = true;
	            }

	            hit = new int[nSel];
	            miss = new int[nSel];
	            for (i=0; i<conjS.length; i++) {
	                for (j=0; j<nClases; j++) {
	                    minDist = Double.POSITIVE_INFINITY;
	                    pos = -1;
	                    for (k=0; k<conjS.length; k++) {
	                        if (i!=k && clasesS[k] == j) {
	                            dist = KNN.distancia(conjS[i], conjR[i], 
	                                    conjN[i], conjM[i], conjS[k], conjR[k], 
	                                    conjN[k], conjM[k], distanceEu);
	                            if (dist < minDist) {
	                                minDist = dist;    						
	                                pos = k;
	                            }
	                        }
	                    }
	                    if (pos >= 0) {
	                        if (clasesS[i] == j) {
	                            hit[pos]++;
	                        } else {
	                            miss[pos]++;
	                        }
	                    }
	                }
	                
	                context.progress();
	            }

	            w = new double[nClases];
	            cc = new int[nClases];
	            for (i=0; i<w.length; i++) {
	                    cont = 0;
	                    for (j=0; j<clasesS.length; j++) {
	                            if (clasesS[j] == i) {
	                                    cont++;
	                            }
	                    }
	                    cc[i] = cont;
	                    w[i] = (double)cont / (double)nSel;
	            }

	            /*RULE H1*/        
	            for (i=0; i<hit.length; i++) {
	                if ((w[clasesS[i]] * (double)miss[i] + epsilon) > 
	                        ((1-w[clasesS[i]]) * (double)hit[i])) {
	                    marcas[i] = false;
	                    nSel--;
	                }
	            }
	            context.progress();

	            /*RULE H3*/
	            if (nClases > 3) {
	                for (i=0; i<hit.length; i++) {
	                    if (!marcas[i] && (miss[i]+hit[i] > 0) 
	                            && miss[i] < (nClases/2)) {
	                        marcas[i] = true;
	                        nSel++;
	                    }
	                }
	            }
	            context.progress();


	            /*RULE H4*/
	            for (i=0; i<hit.length; i++) {
	                if (!marcas[i] && hit[i] >= (cc[clasesS[i]] / 4)) {
	                    marcas[i] = true;
	                    nSel++;
	                }
	            }
	            context.progress();

	            /*
	             * RULE H2_Imb.
	             * Determine IR in the new S. 
	             * When it is higher than that of the previous set, this wlll be 
	             * resolved by adding additional elements in decreasing order of 
	             * their indegree.
	             * Positive class is also not allowed to become the absolute majority.
	             */
	            int nPosS = 0;
	            int nNegS = 0;
	            for(i = 0; i < hit.length; i++){
	                if(marcas[i]){
	                    if(clasesS[i] == posClass){
	                        nPosS++;
	                    } else {
	                        nNegS++;  
	                    }
	                }
	            }

	            if(nPosS == 0 && nNegS == 0){ // everything removed

	                // Select one random element of each class and stop after this iteration
	                int posi;
	                do {
	                    posi = Randomize.Randint(0,hit.length-1);
	                } while (clasesS[posi] != posClass);

	                int nega;
	                do {
	                    nega = Randomize.Randint(0,hit.length-1);
	                } while (clasesS[nega] == posClass);


	                marcas[posi] = true;
	                marcas[nega] = true;

	                nSel = 2;
	                stop = true;
	                
	                context.progress();

	            } else if(nPosS > nNegS){ 
	                // Some negative instances need to be reselected, until IR = 1.
	                scores = new Referencia[hit.length]; 

	                for(i=0; i < hit.length; i++){
	                    scores[i] = new Referencia(i, hit[i] + miss[i]);
	                }

	                Arrays.sort(scores);

	                // Undo marks of negative instances until nNegS == nPosS
	                int c = 0;
	                while(nNegS < nPosS && c < scores.length){
	                    if(clasesS[scores[c].entero] != posClass 
	                            && !marcas[scores[c].entero]){
	                        marcas[scores[c].entero] = true;
	                        nSel++;
	                        nNegS++;
	                    }
	                    c++;
	                }
	                
	                context.progress();


	            } else if(nPosS == 0 || ((double) nNegS / nPosS) > origIR){
	                // Some positive instances need to be reselected
	                scores = new Referencia[hit.length]; 

	                for(i=0; i < hit.length; i++){
	                    scores[i] = new Referencia(i, hit[i] + miss[i]);
	                }

	                Arrays.sort(scores);

	                // Undo marks of negative instances until nNegS == nPosS
	                int c = 0;
	                while(((double) nNegS / nPosS) > origIR && c < scores.length){
	                    if(clasesS[scores[c].entero] == posClass 
	                            && !marcas[scores[c].entero]){
	                        marcas[scores[c].entero] = true;
	                        nSel++;
	                        nPosS++;
	                    }
	                    c++;
	                }
	                
	                context.progress();
	            }  


	            /* The new prototype set */
	            conjS2 = new double[nSel][datosTrain[0].length];
	            conjR2 = new double[nSel][datosTrain[0].length];
	            conjN2 = new int[nSel][datosTrain[0].length];
	            conjM2 = new boolean[nSel][datosTrain[0].length];
	            clasesS2 = new int[nSel];
	            for (m=0, l=0; m<conjS.length; m++) {
	                if (marcas[m]) { //the instance will be evaluated
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS2[l][j] = conjS[m][j];
	                            conjR2[l][j] = conjR[m][j];
	                            conjN2[l][j] = conjN[m][j];
	                            conjM2[l][j] = conjM[m][j];
	                    }
	                    clasesS2[l] = clasesS[m];
	                    l++;
	                }
	            }
	            context.progress();

	            aciertoAct = 0;
	            for (i=0; i<datosTrain.length; i++) {
	                    claseObt = KNN.evaluacionKNN2(1, conjS2, conjR2, conjN2, 
	                            conjM2, clasesS2, datosTrain[i], realTrain[i],
	                            nominalTrain[i], nulosTrain[i], nClases, distanceEu);
	                    if (claseObt == clasesTrain[i]) {
	                            aciertoAct++;
	                    }
	            }
	        } while (aciertoAct >= acierto && nSel < seleccionadosAnt && !stop);
	    } else {
	         do {
	            gmean = gmeanAct;
	            seleccionadosAnt = nSel;

	            /* Copy of the previous set S. */
	            conjS = new double[nSel][datosTrain[0].length];
	            conjR = new double[nSel][datosTrain[0].length];
	            conjN = new int[nSel][datosTrain[0].length];
	            conjM = new boolean[nSel][datosTrain[0].length];
	            clasesS = new int[nSel];
	            for (m=0, l=0; m<nSel; m++) {
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS[l][j] = conjS2[m][j];
	                            conjR[l][j] = conjR2[m][j];
	                            conjN[l][j] = conjN2[m][j];
	                            conjM[l][j] = conjM2[m][j];
	                    }
	                    clasesS[l] = clasesS2[m];
	                    l++;
	            }
	            context.progress();

	            /*Inicialization of the flagged instances vector from the S*/
	            marcas = new boolean[nSel];
	            for (i=0; i<nSel; i++) {
	                    marcas[i] = true;
	            }

	            hit = new int[nSel];
	            miss = new int[nSel];
	            for (i=0; i<conjS.length; i++) {
	                for (j=0; j<nClases; j++) {
	                    minDist = Double.POSITIVE_INFINITY;
	                    pos = -1;
	                    for (k=0; k<conjS.length; k++) {
	                        if (i!=k && clasesS[k] == j) {
	                            dist = KNN.distancia(conjS[i], conjR[i], 
	                                    conjN[i], conjM[i], conjS[k], conjR[k], 
	                                    conjN[k], conjM[k], distanceEu);
	                            if (dist < minDist) {
	                                minDist = dist;    						
	                                pos = k;
	                            }
	                        }
	                    }
	                    if (pos >= 0) {
	                        if (clasesS[i] == j) {
	                            hit[pos]++;
	                        } else {
	                            miss[pos]++;
	                        }
	                    }
	                }
	                context.progress();
	            }

	            w = new double[nClases];
	            cc = new int[nClases];
	            for (i=0; i<w.length; i++) {
	                    cont = 0;
	                    for (j=0; j<clasesS.length; j++) {
	                            if (clasesS[j] == i) {
	                                    cont++;
	                            }
	                    }
	                    cc[i] = cont;
	                    w[i] = (double)cont / (double)nSel;
	            }

	            /*RULE R1*/
	            for (i=0; i<hit.length; i++) {
	                    if ((w[clasesS[i]] * (double)miss[i] + epsilon) > 
	                            ((1-w[clasesS[i]]) * (double)hit[i])) {
	                            marcas[i] = false;
	                            nSel--;
	                    }
	            }
	            context.progress();

	            /*RULE R2*/
	            for (i=0; i<nClases; i++) {
	                    cont = 0;
	                    for (j=0; j<hit.length && cont < 4; j++) {
	                            if (clasesS[j] == i && marcas[j]) {
	                                    cont++;
	                            }
	                    }
	                    if (cont < 4) {
	                            for (j=0; j<hit.length; j++) {
	                                    if (clasesS[j] == i && !marcas[j] 
	                                            && (hit[j]+miss[j]) > 0) {
	                                            marcas[j] = true;
	                                            nSel++;
	                                    }
	                            }
	                    }
	            }
	            context.progress();

	            /*RULE R3*/
	            if (nClases > 3) {
	                    for (i=0; i<hit.length; i++) {
	                            if (!marcas[i] && (miss[i]+hit[i] > 0) 
	                                    && miss[i] < (nClases/2)) {
	                                    marcas[i] = true;
	                                    nSel++;
	                            }
	                    }
	            }
	            context.progress();


	            /*RULE R4*/
	            for (i=0; i<hit.length; i++) {
	                    if (!marcas[i] && hit[i] >= (cc[clasesS[i]] / 4)) {
	                            marcas[i] = true;
	                            nSel++;
	                    }
	            }

	            /* The new prototype set */
	            conjS2 = new double[nSel][datosTrain[0].length];
	            conjR2 = new double[nSel][datosTrain[0].length];
	            conjN2 = new int[nSel][datosTrain[0].length];
	            conjM2 = new boolean[nSel][datosTrain[0].length];
	            clasesS2 = new int[nSel];
	            for (m=0, l=0; m<conjS.length; m++) {
	                if (marcas[m]) { //the instance will be evaluated
	                    for (j=0; j<datosTrain[0].length; j++) {
	                            conjS2[l][j] = conjS[m][j];
	                            conjR2[l][j] = conjR[m][j];
	                            conjN2[l][j] = conjN[m][j];
	                            conjM2[l][j] = conjM[m][j];
	                    }
	                    clasesS2[l] = clasesS[m];
	                    l++;
	                }
	            }
	            context.progress();

	            // Determine gmean of the new set
	            int tn = 0;
	            int tp = 0;
	            for (i=0; i<datosTrain.length; i++) {
	                claseObt = KNN.evaluacionKNN2(1, conjS2, conjR2, conjN2, 
	                        conjM2, clasesS2, datosTrain[i], realTrain[i],
	                        nominalTrain[i], nulosTrain[i], nClases, distanceEu);
	                if (claseObt == clasesTrain[i]) {  // Classified correctly
	                    if(clasesTrain[i] == orig_posClass){ // true positive
	                        tp++;
	                    } else { // true negative
	                        tn++;
	                    }
	                }
	            }
	            double tpr = (double) tp / nPos;
	            double tnr = (double) tn / nNeg;
	            gmeanAct = Math.sqrt(tpr * tnr);
	        } while (gmeanAct >= gmean && nSel < seleccionadosAnt);
	        
	    }

	    System.out.println("HMNEI_Imb "+ relation + " " 
	            + (double)(System.currentTimeMillis()-tiempo)/1000.0 + "s");

	    OutputIS.escribeSalida(ficheroSalida[0], conjR, conjN, conjM, clasesS, 
	            entradas, salida, nEntradas, relation);
	    OutputIS.escribeSalida(ficheroSalida[1], test, entradas, salida, 
	            nEntradas, relation);
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
	    	
	    	this.epsilon = 0.1;
	    	this.distanceEu = true;
			this.version = 1;				
	    	
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
	
		    /*Getting epsilon value*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    epsilon = Double.parseDouble(tokens.nextToken().substring(1));
	
		    /*Getting the type of distance function*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    distanceEu = tokens.nextToken().substring(1).equalsIgnoreCase("Euclidean")?true:false;
		    
		    /*Perform HMNEI_Imb1, HMNEI_Imb2 or HMNEI_Imb3*/
		    linea = lineasFichero.nextToken();
		    tokens = new StringTokenizer (linea, "=");
		    tokens.nextToken();
		    version = Integer.parseInt(tokens.nextToken().substring(1));
	    }
	  }
}
