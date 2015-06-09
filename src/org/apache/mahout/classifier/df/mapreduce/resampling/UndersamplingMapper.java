package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.UndersamplingBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UndersamplingMapper extends Mapper<LongWritable, Text, IntWritable, Text>{

  private static final Logger log = LoggerFactory.getLogger(UndersamplingMapper.class);
  private Dataset dataset;
  boolean noOutput;
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  protected void setup(Context context) throws IOException, InterruptedException {
	     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !UndersamplingBuilder.isOutput(conf);
    
    dataset = UndersamplingBuilder.loadDataset(conf);
    
    converter = new DataConverter(dataset); 
  }      
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    
    Instance instance = converter.convert(value.toString());	   
    int label = (int)dataset.getLabel(instance);        
    IntWritable id = new IntWritable(label);     
	context.write(id, value);   
	
  }
	  
}
