package org.apache.mahout.classifier.pg.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.pg.builder.PGgenerator;
import org.apache.mahout.classifier.pg.data.Dataset;
import org.apache.mahout.classifier.pg.mapreduce.partial.StrataID;
import org.apache.mahout.classifier.pg.utils.PGUtils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Distance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;

import java.io.IOException;
import java.util.Arrays;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class FilteringIterativeReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected PGgenerator pg_algorithm;
  
  private Dataset dataset;
  protected String header;

  protected PrototypeSet join = new PrototypeSet();
  protected int strata;
  private int firstId = 0;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected PGgenerator getPGgeneratorBuilder() {
    return pg_algorithm;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getPGgeneratorBuilder(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, PGgenerator pg_algorithm, Dataset dataset, String header) {
    Preconditions.checkArgument(pg_algorithm != null, "PGgenerator not found in the Job parameters");
    this.noOutput = noOutput;
    this.pg_algorithm = pg_algorithm;
    this.dataset = dataset;
    this.header = header;
  }

  
  /**
   * Generic reducer, it only adds all the RSs.
   */
  
protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context)
		throws IOException, InterruptedException {
	// TODO Apéndice de método generado automáticamente
	
	//System.out.println("Si paso por aquí: "+id);
	//strata = (StrataID) id;

	for(VALUEIN value: rs){
		MapredOutput prueba = (MapredOutput) value;
		PrototypeSet strato = prueba.getRS();
		
		context.progress();
		PGUtils.readHeader(this.header);
	    PrototypeSet filtered = new PrototypeSet(ENN(strato, context));
	    context.progress();
	    
	    //if(filtered.size()>0)
	       join.add(filtered);
	    //else
	    //	join.add(strato);
	    
    	System.out.println("Resultados: "+join.size()+","+strato.size()+","+filtered.size());
    	
		// if you write here, the cleanup does not work.
	//	MapredOutput salida= new MapredOutput(join);
	//	context.write((KEYOUT) id, (VALUEOUT) salida);
	}
	
	System.out.println("*******************");


}


protected void cleanup(Context context) throws IOException, InterruptedException {
	 
    System.out.println("escribo el join.");
    StrataID key = new StrataID();

    key.set(strata, firstId + 1);
    
	MapredOutput salida= new MapredOutput(join);
	context.write((KEYOUT) key, (VALUEOUT) salida);
}



/**
 * 
 * Edited nearest neighbor of T. , PrototypeSet labeled
 * @return
 */
public PrototypeSet ENN (PrototypeSet T, Context context)
{
	//T.print();
	 PrototypeSet Sew = new PrototypeSet (T);
	
	 //this.k = 7;
	  // Elimination rule kohonen
	  
	 // System.out.println("Mayor�a " + majority);


	  int toClean[] = new int [T.size()];
	  Arrays.fill(toClean, 0);
	  int pos = 0;
	  
	  // computing majority per class. It is possible that there is no representatives
	  // from all the classes.
	  
	  int numberOfClasses = T.getPosibleValuesOfOutput().size();
	  System.out.println("number of classes "+ numberOfClasses);
	  
	  int majority[] = new int[numberOfClasses];
	  
	  for(int i=0; i< numberOfClasses; i++){
		  if(T.getFromClass(i).size()<3){
			  majority[i] = 0;   // never remove
		  }else
			  majority[i] = 3/2 + 1;
	  }
	  
	for ( Prototype p : T){
		context.progress();
		 double class_p = p.getOutput(0);
			 
  		 Distance.setNumberOfInputs(p.numberOfInputs());
		 PrototypeSet neighbors = KNN.knn(p, T, 3); // labeled
		
		int counter= 0;
		  for(Prototype q1 :neighbors ){
			double class_q1 = q1.getOutput(0);
			
			if(class_q1 == class_p){
				counter++;
			} 
			
		  }
		  
		  //System.out.println("Misma clase = "+ counter);
		  if ( counter < majority[(int) class_p]){ // We must eliminate this prototype.
			  toClean [pos] = 1; // we will clean			  
		  }
		   pos++;
	}
	
	//Clean the prototypes.
	PrototypeSet aux= new PrototypeSet();
	for(int i= 0; i< toClean.length;i++){
		if(toClean[i] == 0)
			aux.add(T.get(i));
		
	}
	//Remove aux prototype set
	
	Sew = aux;
	
	//System.out.println("Result of filtering");	
	//Sew.print();

	return Sew;
	  
}


}


