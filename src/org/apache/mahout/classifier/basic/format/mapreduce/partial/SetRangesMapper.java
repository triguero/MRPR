package org.apache.mahout.classifier.basic.format.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.basic.format.mapreduce.rangesOutput;
import org.apache.mahout.classifier.basic.format.mapreduce.Builder;
import org.apache.mahout.classifier.basic.format.mapreduce.MapredMapper;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.DataConverter;
import org.apache.mahout.classifier.basic.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Pair;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class SetRangesMapper extends MapredMapper<LongWritable,Text,StrataID,rangesOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(SetRangesMapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  private double[] mins;
  private double[] maxs;
  
  
  public int getFirstTreeId() {
    return firstId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    context.progress();
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), Builder.getHeader(conf));
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
  protected void configure(int partition, int numMapTasks, String header) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    this.header=header;

    log.debug("partition : {}", partition);
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));
    context.progress();
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // prepare the data
    log.debug("partition: {} numInstances: {}", partition, instances.size());
        
    context.progress();

    Data data = new Data(getDataset(), instances);
    
    log.info("SetRanges: Size of data partition= "+data.size());
    
    mins=new double[data.get(0).getInstance().size()-1];
    Arrays.fill(mins, Double.MAX_VALUE);
    maxs=new double[data.get(0).getInstance().size()-1];
    Arrays.fill(maxs, Double.MIN_VALUE);
    
    
    for(int i=0; i<data.size();i++){
    	
    	for(int j=0;j<(data.get(i).get().length-1);j++){
    	
	    	if(data.get(i).get()[j]<mins[j]){
	    		mins[j]=data.get(i).get()[j];
	  	    }
	    	if(data.get(i).get()[j]>maxs[j]){
	    		maxs[j]=data.get(i).get()[j];
	    	}
    	
    	}
    }
    
  /*  for(int j=0;j<(data.get(0).get().length-1);j++){
    	System.out.println(mins[j]+", "+maxs[j]);
    }*/
    context.progress();

            
    StrataID key = new StrataID();

    key.set(partition, firstId + 1);
      
    rangesOutput emOut = new rangesOutput(mins,maxs);
      
    context.write(key, emOut);
   
  }
}
