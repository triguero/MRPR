package org.apache.mahout.classifier.smo.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.smo.builder.SMOgenerator;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.smo.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class MajorityIterativeReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected SMOgenerator smo_algorithm;
  
  private Dataset dataset;
  protected String header;

  protected int Majority[][]=null;
  protected int strata;
  private int firstId = 0;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected SMOgenerator getPGgeneratorBuilder() {
    return smo_algorithm;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getSMOgeneratorBuilder(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, SMOgenerator smo_algorithm, Dataset dataset, String header) {
    Preconditions.checkArgument(smo_algorithm != null, "PGgenerator not found in the Job parameters");
    this.noOutput = noOutput;
    this.smo_algorithm = smo_algorithm;
    this.dataset = dataset;
    this.header = header;
  }

  
	  /**
	   * Generic reducer, it only adds all the RSs.
	   */
  
	protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context)
			throws IOException, InterruptedException {
		// TODO Apéndice de método generado automáticamente
	
		System.out.println("Si paso por aquí: "+id);
		//strata = (StrataID) id;

		for(VALUEIN value: rs){
			MapredOutput prueba = (MapredOutput) value;

			ArrayList<Integer>  clasificacion= prueba.getPredictions();
			
			
			for(int i=0;i<clasificacion.size();i++){
				System.out.print(clasificacion.get(i)+",");
			}
			
			if(Majority==null){
				Majority=new int[clasificacion.size()][prueba.getNumClases()];
				for(int i=0;i<clasificacion.size();i++)
				    Arrays.fill(Majority[i], 0);
				
			}
			
			for(int i=0;i<clasificacion.size();i++){
				if(clasificacion.get(i)!=Integer.MIN_VALUE)
					Majority[i][clasificacion.get(i)]++;
			}
			
			
			context.progress();
			
	    	
			// if you write here, the cleanup does not work.
			
			//MapredOutput salida= new MapredOutput(Majority);
			//context.write((KEYOUT) id, (VALUEOUT) salida);
			
			//System.out.println("Dentro");
		}
		// System.out.println("just Majority");


	
	}


	 protected void cleanup(Context context) throws IOException, InterruptedException {
		 
		    System.out.println("escribo la mayoría.");
		    StrataID key = new StrataID();

		    key.set(strata, firstId + 1);
		    
		    ArrayList<Integer> pre = new ArrayList<Integer>();// new int[this.Majority.length];
		    
		    // calcular el maximo de cada.
		    
		    for(int i=0; i<Majority.length; i++){   // recorre instancias
		    	
		    	double max = Majority[i][0];
		    	pre.add(0);
		    	for (int j=1; j<Majority[0].length; j++){  // recorre clases
		    		if(Majority[i][j]>max){
		    			max=Majority[i][j];
		    			pre.set(i,j);
		    		}
		    	}
		    }
		    
		    System.out.println("CLEAN-UP");
			for(int i=0;i<pre.size();i++){
				System.out.print(pre.get(i)+",");
			}
			
		    
			MapredOutput salida= new MapredOutput(pre,Majority[0].length);
			context.write((KEYOUT) key, (VALUEOUT) salida);
	 }
}


