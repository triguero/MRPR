package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.UndersamplingBuilder;    
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
 
public class UndersamplingReducer extends Reducer<IntWritable, Text, NullWritable, Text>{
	
  private static final Logger log = LoggerFactory.getLogger(UndersamplingReducer.class);
  private Dataset dataset;
  private Data data;
  private DataConverter converter;
  boolean noOutput;
  
  private int negativeClass;
  private int positiveClass;
  private int classes_distribution [];
  private List<Instance> final_instances = new ArrayList<Instance>();
  
  /**
  * Load the training data
  */
   private static Data loadData(Configuration conf, Dataset dataset) throws IOException {
     Path dataPath = UndersamplingBuilder.getDistributedCacheFile(conf, 1);
     FileSystem fs = FileSystem.get(dataPath.toUri(), conf);
     return DataLoader.loadData(dataset, fs, dataPath);
   }
  	
   protected void setup(Context context) throws IOException, InterruptedException {
	     
     super.setup(context);
   
     Configuration conf = context.getConfiguration();
   
     log.info("Loading the data...");
   
     dataset = UndersamplingBuilder.loadDataset(conf);
   
     data = loadData(conf, dataset);
   
     log.info("Data loaded : {} instances", data.size());
   
     classes_distribution = data.computeClassDistribution();
   
     negativeClass = data.computeNegativeClass(classes_distribution);  
     
     positiveClass = data.computePositiveClass(classes_distribution);
     
     converter = new DataConverter(dataset);   
   }
  
  
   public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {	 
	 if(key.get() == negativeClass){
	   List<Instance> instances = new ArrayList<Instance>();
       for (Text value : values) { 
    	 Instance instance = converter.convert(value.toString());
         instances.add(instance);   
       }
       java.util.Collections.shuffle(instances);       
       for (Instance instance : instances.subList(0, classes_distribution[positiveClass])) {
         final_instances.add(instance);
       }
     }
     else{       
       for (Text value : values) { 
    	 Instance instance = converter.convert(value.toString());
    	 final_instances.add(instance);
	   }
     }
   }
   
   
   protected void cleanup (Context context) throws IOException, InterruptedException{
	 NullWritable id = null;  
	 Data newData = new Data(dataset, final_instances);
	   
	 newData.randomizeData();
	   
	 List<Instance> randomizeInstances = newData.getInstances(); 
	 
	 for (int i = 0 ; i < randomizeInstances.size() ; i++){
       StringBuilder returnString = new StringBuilder();
 	   returnString.append(randomizeInstances.get(i).toString(dataset)).append(dataset.getLabelString(dataset.getLabel(randomizeInstances.get(i))));  
       String instance = returnString.toString();       
 	   Text text = new Text(instance);
       context.write(id, text);
     }
	   
   }

}
