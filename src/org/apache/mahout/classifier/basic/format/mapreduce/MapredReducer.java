package org.apache.mahout.classifier.basic.format.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.basic.format.mapreduce.partial.StrataID;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  
  private Dataset dataset;
  protected String header;

  protected PrototypeSet join = new PrototypeSet();
  protected int strata;

  
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
    
    configure(!Builder.isOutput(conf),  Builder.loadDataset(conf), Builder.getHeader(conf));
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
	

}


}


