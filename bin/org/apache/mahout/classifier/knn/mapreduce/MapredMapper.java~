package org.apache.mahout.classifier.chi_rw.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.chi_rw.builder.Fuzzy_ChiBuilder;
import org.apache.mahout.classifier.chi_rw.data.Dataset;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Mapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected Fuzzy_ChiBuilder fuzzy_ChiBuilder;
  
  private Dataset dataset;
  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected Fuzzy_ChiBuilder getFuzzy_ChiCSBuilder() {
    return fuzzy_ChiBuilder;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getFuzzy_ChiBuilder(conf), Builder.loadDataset(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, Fuzzy_ChiBuilder fuzzy_ChiBuilder, Dataset dataset) {
    Preconditions.checkArgument(fuzzy_ChiBuilder != null, "Fuzzy_ChiCSBuilder not found in the Job parameters");
    this.noOutput = noOutput;
    this.fuzzy_ChiBuilder = fuzzy_ChiBuilder;
    this.dataset = dataset;
  }
}

