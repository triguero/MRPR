package org.apache.mahout.classifier.smo.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.feature_selection.mapreduce.MajorityIterativeReducer;
import org.apache.mahout.classifier.feature_selection.mapreduce.MapredOutput;
import org.apache.mahout.classifier.feature_selection.mapreduce.Builder;
import org.apache.mahout.classifier.feature_selection.mapreduce.MapredReducer;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.DataConverter;
import org.apache.mahout.classifier.basic.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class MajorityReducer extends MajorityIterativeReducer<StrataID,MapredOutput,StrataID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(MajorityReducer.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
 
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
   

  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    log.info("Configuring reducer");
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
  */
  protected void configure(int partition, int numMapTasks) {
   // converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.strata = partition;
    
    log.debug("partition : {}", partition);
  }
   

}
