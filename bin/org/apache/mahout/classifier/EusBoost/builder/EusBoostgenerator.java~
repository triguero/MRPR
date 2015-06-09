package org.apache.mahout.classifier.smo.builder;

import java.util.ArrayList;

import org.apache.mahout.classifier.basic.utils.HandlerCSVM;
import  org.apache.mahout.classifier.basic.utils.HandlerSMO;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.classifier.smo.*;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.classifier.basic.utils.C45.C45;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SMOgenerator  {
  
  private static final Logger log = LoggerFactory.getLogger(SMOgenerator.class);	
  int nClasses, nLabels;

  public String SMOmethod = "CHC";
  
  public String header;
  
  HandlerSMO SMO;
  
  HandlerCSVM cSVM;
  
  C45 c45;
  
  ArrayList<Integer> predicciones;
  
	//  strata[i].print();
	  
  public SMOgenerator() {
  }
  
  public SMOgenerator(String alg)
  {
	  this.SMOmethod = alg;
  }
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }

  public void setHeader(String header){
	  this.header=header;
  }
  
  public void build(Data data, FileSystem fs, Path test, String header, Context context, int partition) throws Exception {
    //We do here the algorithm's operations

	Dataset dataset = data.getDataset();
	 
	nClasses = dataset.nblabels();
	
	context.progress();
	
	log.info("SMOgenerator: ejecutando SMO...");

	PrototypeSet train = new PrototypeSet(data,context);

    Utils.readHeader(header);

    //PrototypeSet testSet= Utils.readTest(fs, test);   
    
        
	//cSVM= new HandlerCSVM(train.toInstanceSet(), testSet.toInstanceSet(), nClasses, String.valueOf("123456"),header);    // CSVM

		
//	SMO= new HandlerSMO(train.toInstanceSet(), testSet.toInstanceSet(), nClasses, String.valueOf("123456"),header);      // SMO
		
    
	//SMO= new HandlerSMO(train.toInstanceSet(), fs, test, nClasses, String.valueOf("123456"),header, context, partition);      // SMO
	  
    
    
	c45  = new C45(train.toInstanceSet(),context);      // C4.5 called
	  
   	log.info("SMO: data size = "+data.size());
	
	predicciones=c45.getPredicciones(fs,test);
	
	  //predicciones= cSVM.getPredicciones();

  // 	predicciones=SMO.getPredicciones();
	
	//log.info("SMO: test size = "+test.size());

  }


  public ArrayList<Integer> getPredictions() {
	  

	  return predicciones;

  }


}
