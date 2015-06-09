package org.apache.mahout.classifier.basic.format.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.basic.format.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;

import java.io.IOException;
import java.util.Arrays;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class MajorityIterativeReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  
  private Dataset dataset;
  protected String header;

  private double[] minsR;
  private double[] maxsR;
  
  protected int mappers=0;
  protected int strata;
  private int firstId = 0;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, Dataset dataset, String header) {
    this.noOutput = noOutput;
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
			rangesOutput prueba = (rangesOutput) value;

	        if (minsR==null) {
	            minsR=new double[prueba.mins.length];
	            Arrays.fill(minsR, Double.MAX_VALUE);
	            maxsR=new double[prueba.mins.length];
	            Arrays.fill(maxsR, Double.MIN_VALUE);
	         }
	        
	        for(int i=0; i<prueba.mins.length;i++){
	        	//System.out.println(prueba.mins[i]+", "+prueba.maxs[i]);
	        	if(prueba.mins[i]<minsR[i]){
	        		minsR[i]=prueba.mins[i];
	      	    }
	        	if(prueba.maxs[i]>maxsR[i]){
	        		maxsR[i]=prueba.maxs[i];
	        	}
	        }
	        
			context.progress();
	

		}



	
	}


	 protected void cleanup(Context context) throws IOException, InterruptedException {
		 
		    StrataID key = new StrataID();

		    key.set(strata, firstId + 1);
		    

			
			rangesOutput salida= new rangesOutput(minsR,maxsR);
			context.write((KEYOUT) key, (VALUEOUT) salida);
	 }
}


