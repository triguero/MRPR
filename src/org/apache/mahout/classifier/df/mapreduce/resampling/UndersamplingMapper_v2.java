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
import org.apache.mahout.classifier.df.mapreduce.UndersamplingBuilder_v2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UndersamplingMapper_v2 extends Mapper<LongWritable, Text, LongWritable, Text>{

  private static final Logger log = LoggerFactory.getLogger(UndersamplingMapper_v2.class);
  private Dataset dataset;
  boolean noOutput;
  
  private int negativeClass, positiveClass;
  private int classes_distribution [];
  private double elimination_factor;
  private int counter = 0;
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  protected void setup(Context context) throws IOException, InterruptedException {
	     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !UndersamplingBuilder_v2.isOutput(conf);
    
    dataset = UndersamplingBuilder_v2.loadDataset(conf);
    
    converter = new DataConverter(dataset); 
    
    // The number of occurrences of each label value		  		     
    classes_distribution = new int [dataset.nblabels()];
    
    //classes_distribution[0]=31305192;
    //classes_distribution[1]=687729;
    
    classes_distribution[0]=100;
    classes_distribution[1]=5;
    
    negativeClass = computeNegativeClass(classes_distribution);  
    positiveClass = computePositiveClass(classes_distribution);
    
    //elimination_factor = (double)classes_distribution[1] / (double)classes_distribution[0]; //50-50
    
    //elimination_factor = 0.5; //50% negativas
    
    elimination_factor = (double)(classes_distribution[1]*3) / (double)classes_distribution[0]; //nº instancias negativas: doble del nº de instancias positivas
    
    //elimination_factor = (classes_distribution[negativeClass]-(classes_distribution[positiveClass]*1.5))/classes_distribution[negativeClass];
	
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
  
  public int computePositiveClass(int classes_distribution []) {	        
	int n_classes = dataset.nblabels();
    int min = classes_distribution[0];
    int pos_min = 0;
    for (int i=1; i<n_classes; i++) {
      if (classes_distribution[i] < min) {
        pos_min = i;
        min = classes_distribution[i];
      }
    }     
    return pos_min;
  }
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {    
    Instance instance = converter.convert(value.toString());	   
    int label = (int)dataset.getLabel(instance);  
    
    Random r = new Random();
    double random;
        
    if (!noOutput) {  		
		if(label == positiveClass){		
    		context.write(key, value);  
		}
		else{//negative class
			random = r.nextDouble(); 
			if(random  < elimination_factor){   
				context.write(key, value);  
			}
		}	
	}    	  
	
  }
  
  
	  
}

