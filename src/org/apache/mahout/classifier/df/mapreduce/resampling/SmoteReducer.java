package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.SmoteBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SmoteReducer extends Reducer<LongWritable, Text, NullWritable, Text>{
	
  private static final Logger log = LoggerFactory.getLogger(SmoteReducer.class);
  private Dataset dataset;
  private DataConverter converter;
	
  protected void setup(Context context) throws IOException, InterruptedException {
     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    dataset = SmoteBuilder.loadDataset(conf);
    
    converter = new DataConverter(dataset);
  } 
  
  public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
	    NullWritable id = null;
	    	    
	    for (Text value : values) {  	    	
	    	context.write(id, value);
	  	}  
  }
  
  /*
  public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
    NullWritable id = null;
    
    List<Instance> instances = new ArrayList<Instance>();
    
    for (Text value : values) {   
      Instance instance = converter.convert(value.toString());	
      instances.add(instance);
  	}
    
    Data newData = new Data(dataset, instances);
    
    log.info("Randomize data...");
    newData.randomizeData();
    
    List<Instance> randomizeInstances = newData.getInstances(); 
    
    log.info("Copying final instances...");
    for (int i = 0 ; i < randomizeInstances.size() ; i++){
      StringBuilder returnString = new StringBuilder();
 	  returnString.append(randomizeInstances.get(i).toString(dataset)).append(dataset.getLabelString(dataset.getLabel(randomizeInstances.get(i))));  
      String instance = returnString.toString();       
 	  Text text = new Text(instance);
      context.write(id, text);
    }
  }
  */
}
