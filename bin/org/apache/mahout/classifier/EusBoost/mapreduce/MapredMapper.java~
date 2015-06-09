package org.apache.mahout.classifier.smo.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.smo.builder.SMOgenerator;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.hadoop.fs.Path;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Mapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected SMOgenerator smo_algorithm;
  
  protected String header;
  private Dataset dataset;
  protected Path testPath;
  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected SMOgenerator getSMOgeneratorBuilder() {
    return smo_algorithm;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  protected String getInstanceSet() {
	return header;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getSMOgeneratorBuilder(conf), Builder.loadDataset(conf), Builder.getHeader(conf), Builder.getTestPath(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, SMOgenerator smo_algorithm, Dataset dataset, String header, String testPath) {
    Preconditions.checkArgument(smo_algorithm != null, "SMOGgenerator not found in the Job parameters");
    this.noOutput = noOutput;
    this.smo_algorithm = smo_algorithm;
    this.dataset = dataset;
    this.header = header;
    this.testPath=new Path(testPath);
  }
}


