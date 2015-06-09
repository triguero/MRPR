package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.HybridBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HybridMapper extends Mapper<LongWritable, Text, LongWritable, Text>{
	
  private static final Logger log = LoggerFactory.getLogger(HybridMapper.class);
  private Dataset dataset;
  boolean noOutput;
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  private int negativeClass;
  private int classes_distribution [];
  private int pos_replication;
  private double elimination_factor;
  
  protected void setup(Context context) throws IOException, InterruptedException {
     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !HybridBuilder.isOutput(conf);
    
    log.info("Loading the data...");
    
    dataset = HybridBuilder.loadDataset(conf);
       
    // The number of occurrences of each label value		  		     
    classes_distribution = new int [dataset.nblabels()];
    
    classes_distribution[0]=31305192;
    classes_distribution[1]=687729;
    
    //classes_distribution[0]=100;
    //classes_distribution[1]=5;
    
    pos_replication = 5;
    
    elimination_factor = 0.5;
    
    negativeClass = computeNegativeClass(classes_distribution);  
    
    converter = new DataConverter(dataset); 
  }      
  
  public int computeNegativeClass(int classes_distribution []) {	        
	int n_classes = dataset.nblabels();
    int max = classes_distribution[0];
    int pos_max = 0;
    for (int i=1; i<n_classes; i++) {
      if (classes_distribution[i] > max) {
        pos_max = i;
        max = classes_distribution[i];
      }
    }     
    return pos_max;
  }
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {	  
    Instance instance = converter.convert(value.toString());    
    int label = (int)dataset.getLabel(instance);
		
	LongWritable id;
	Random r = new Random();		  
	double random;
	int random_;
	
	if (!noOutput) {//negative class		
		if(label == negativeClass){									
    		random = r.nextDouble(); 
			if(random  < elimination_factor){   
				random_ = r.nextInt(pos_replication); 
	    		id = new LongWritable(random_);
				context.write(id, value);  
			}
		}
		else{//positive class
			for(int i = 0 ; i < pos_replication ; i++){   
				id = new LongWritable(i);
				context.write(id, value);  
			}
		}	
	}    	  
  }
    
}
