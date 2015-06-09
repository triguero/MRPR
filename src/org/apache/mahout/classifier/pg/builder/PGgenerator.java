package org.apache.mahout.classifier.pg.builder;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerSSMASFLSDE;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.HandlerIS;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.classifier.pg.data.Data;
import org.apache.mahout.classifier.pg.data.Dataset;
import org.apache.mahout.classifier.pg.utils.PGUtils;
import org.apache.mahout.classifier.pg.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ICPL.ICPLGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDE.IPLDEGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS.IPLDECSGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS_windowing.IPLDECS_windowingGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDE_windowing.IPLDE_windowingGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.IPADE_NB.IPADE_NBGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.LVQ.LVQ3;
import org.apache.mahout.keel.Algorithms.Instance_Generation.RSP.RSPGenerator;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PGgenerator  {
  
  private static final Logger log = LoggerFactory.getLogger(PGgenerator.class);	
  int nClasses, nLabels;

  public String PGmethod = "IPADE";
  
  public String header;
  IPLDEGenerator algorithm;
  HandlerSSMASFLSDE algorithm2;
  IPLDECSGenerator algorithmIPADECS;

  IPLDE_windowingGenerator algorithmIPADEwin;
  IPLDECS_windowingGenerator algorithmIPADECSwin;
  IPADE_NBGenerator algorithmIPADE_NB;
  ICPLGenerator algorithmICPL;
  LVQ3 algorithmLVQ3;
  RSPGenerator algorithmRSP3;
  
  HandlerIS algorithmIS;
  
  
	//  strata[i].print();
	  
  public PGgenerator() {
  }
  
  public PGgenerator(String alg)
  {
	  this.PGmethod = alg;
  }
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }


  public void setHeader(String header){
	  this.header=header;
  }
  public void build(Data data, Context context, int windows) throws Exception {
    //We do here the algorithm's operations

	Dataset dataset = data.getDataset();
	 
	nClasses = dataset.nblabels();
	
	//Gets the number of input attributes of the data-set
	int nInputs = dataset.nbAttributes() - 1;
	
	//It returns the class labels
	String clases[] = dataset.labels();
	
	// data has the instance + label
	//for(int i=0; i<= nInputs; i++)
	//	log.info("prueba: "+data.get(0).get()[i]);
	context.progress();

	if(this.PGmethod.equalsIgnoreCase("IPADE")){
		log.info("PGgenerator: ejecutando IPADE...");
		algorithm = new IPLDEGenerator(context, data, 1, 10000, 8, 20, 0.5, 0.9, 0.03, 0.07);
	}else if(this.PGmethod.equalsIgnoreCase("SSMASFLSDE")){
		log.info("PGgenerator: ejecutando SSMASFLSDE...");
		algorithm2 = new HandlerSSMASFLSDE();
		algorithm2.ejecutar(data, context);
	}else if(this.PGmethod.equalsIgnoreCase("IPADECS")){
		log.info("PGgenerator: ejecutando IPADECS...");
		algorithmIPADECS = new IPLDECSGenerator(context, data, 1, 10, 500, 8, 20, 0.1, 0.9, 0.1,0.1,  0.03, 0.07, 3);
	}else if(this.PGmethod.equalsIgnoreCase("IPADE_windowing")){	
		log.info("PGgenerator: ejecutando IPADE con windowing...");
		algorithmIPADEwin = new IPLDE_windowingGenerator(context, data, 1, 10000, 8, 20, 0.5, 0.9, 0.03, 0.07,windows); // cambiar por número de stratos.
	}else if(this.PGmethod.equalsIgnoreCase("IPADECS_windowing")){	
		log.info("PGgenerator: ejecutando IPADECS con windowing "+ windows);
		algorithmIPADECSwin = new IPLDECS_windowingGenerator(context, data, 1, 10, 500, 8, 20, 0.1, 0.9, 0.1,0.1,  0.03, 0.07, 3, windows);
	}else if(this.PGmethod.equalsIgnoreCase("LVQ3")){
		log.info("PGgenerator: ejecutando LVQ3 ");
		algorithmLVQ3 = new LVQ3(context,data, 100, 2, 0.1, 0.2, 0.1);
	}else if(this.PGmethod.equalsIgnoreCase("RSP3")){
		log.info("PGgenerator: ejecutando RSP3 ");
		algorithmRSP3 = new RSPGenerator(data, 0,"diameter", context);
	}else if(this.PGmethod.equalsIgnoreCase("DROP3") || this.PGmethod.equalsIgnoreCase("FCNN")){
		log.info("PGgenerator: ejecutando "+this.PGmethod);
		
		algorithmIS = new HandlerIS(this.PGmethod);
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);
		
	}else if(this.PGmethod.equalsIgnoreCase("IPADE_NB_NN")){
		log.info("PGgenerator: ejecutando IPADE_NB ");
		algorithmIPADE_NB=new IPADE_NBGenerator(context,data, 1,1000,8,20,0.5, 0.9, 0.03, 0.07,"NN","false",0.5);
	}else if(this.PGmethod.equalsIgnoreCase("IPADE_NB_C45")){
		log.info("PGgenerator: ejecutando IPADE_NB ");
		algorithmIPADE_NB=new IPADE_NBGenerator(context,data, 1,1000,8,20,0.5, 0.9, 0.03, 0.07,"C45","false",0.5);
	}else if(this.PGmethod.equalsIgnoreCase("SSMA_Imb")){ // SARAH
		log.info("PGgenerator: ejecutando SSMA_Imb");
		algorithmIS = new HandlerIS("SSMAImb");
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);			
	} else if(this.PGmethod.equalsIgnoreCase("SSMA_Imb_W")){ // SARAH
		log.info("PGgenerator: ejecutando SSMA_Imb with windowing");
		algorithmIS = new HandlerIS("SSMAImb-W");
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);			
	} else if(this.PGmethod.equalsIgnoreCase("ENNTh_Imb")){ // SARAH
		log.info("PGgenerator: ejecutando ENNTh_Imb");
		algorithmIS = new HandlerIS("ENNThImb");
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);
	}else if(this.PGmethod.equalsIgnoreCase("HMNEI_Imb")){ // SARAH
		log.info("PGgenerator: ejecutando HMNEI_Imb");
		algorithmIS = new HandlerIS("HMNEIImb");
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);
	}else if(this.PGmethod.equalsIgnoreCase("Reconsistent_Imb")){ // SARAH
		log.info("PGgenerator: ejecutando Reconsistent_Imb");
		algorithmIS = new HandlerIS("ReconsImb");
		algorithmIS.reduceSet = algorithmIS.reduce(data, context);
	}else {
		log.info("PGgenerator: No hay reducción, guardo el fichero de entrada tal cual.");
		algorithm2 = new HandlerSSMASFLSDE();
		algorithm2.reducedSet = new PrototypeSet(data, context);
	}

	log.info("PG: data size = "+data.size());

  }
  /*else if(this.PGmethod.equalsIgnoreCase("ICPL")){
	log.info("PGgenerator: ejecutando ICPL2 ");  
	algorithmICPL= new ICPLGenerator(data,2,"RT2",1,0, context); 

}*/

  public PrototypeSet reduceSet() {
	  PrototypeSet output=null;
	  
	  if(this.PGmethod.equalsIgnoreCase("IPADE")){
		 output=algorithm.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("IPADECS")){
		 output=algorithmIPADECS.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("IPADE_windowing")){
		 output=algorithmIPADEwin.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("IPADECS_windowing")){
		 output=algorithmIPADECSwin.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("ICPL")){
   	     output=algorithmICPL.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("LVQ3")){
	   	 output=algorithmLVQ3.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("RSP3")){
	   	 output=algorithmRSP3.reduceSet();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("DROP3") || this.PGmethod.equalsIgnoreCase("FCNN")){
		 output = algorithmIS.reduceSet;
	  }else if(this.PGmethod.equalsIgnoreCase("SSMA_Imb")
			  || this.PGmethod.equalsIgnoreCase("SSMA_Imb_W") 
			  || this.PGmethod.equalsIgnoreCase("ENNTh_Imb") 
			  || this.PGmethod.equalsIgnoreCase("HMNEI_Imb") 
			  || this.PGmethod.equalsIgnoreCase("Reconsistent_Imb") ){
		 output = algorithmIS.reduceSet;
	  }else if(this.PGmethod.equalsIgnoreCase("IPADE_NB_NN")){
	   	 output=algorithmIPADE_NB.reduceSetNB();
		 output.applyThresholds();
	  }else if(this.PGmethod.equalsIgnoreCase("IPADE_NB_C45")){
	   	 output=algorithmIPADE_NB.reduceSetNB();
		 output.applyThresholds();
	  }else{
		  output=algorithm2.reducedSet;
	  }
	  
	  log.info("PG: RS size = "+output.size());
   
	  //log.info("\n"+output.toString());
	  return output;
  }


}
