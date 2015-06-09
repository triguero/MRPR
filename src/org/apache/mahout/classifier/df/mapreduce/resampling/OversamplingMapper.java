package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.OversamplingBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class OversamplingMapper extends Mapper<LongWritable, Text, LongWritable, Text>{
  
  private static final Logger log = LoggerFactory.getLogger(OversamplingMapper.class);
  private Dataset dataset;
  boolean noOutput;
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  private int negativeClass;
  private int classes_distribution [];
  private int replication;
  
  protected void setup(Context context) throws IOException, InterruptedException {
     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !OversamplingBuilder.isOutput(conf);
    
    log.info("Loading the data...");
    
    dataset = OversamplingBuilder.loadDataset(conf);
       
    // The number of occurrences of each label value		  		     
    classes_distribution = new int [dataset.nblabels()];
    
    if(OversamplingBuilder.getNbNeg(conf) > OversamplingBuilder.getNbPos(conf)){    
      classes_distribution[0]= OversamplingBuilder.getNbNeg(conf);
      classes_distribution[1]= OversamplingBuilder.getNbPos(conf);
    }
    else{
	  classes_distribution[0]= OversamplingBuilder.getNbPos(conf);
      classes_distribution[1]= OversamplingBuilder.getNbNeg(conf);
    }
          
    negativeClass = OversamplingBuilder.getNegClass(conf);
    
    converter = new DataConverter(dataset); 
    
    double factor = (classes_distribution[0] / classes_distribution[1]); //ROS100
    
   // double factor = ((classes_distribution[0] + classes_distribution[0]) / classes_distribution[1]); //ROS200
    
    //double factor = ((classes_distribution[1]*2) / classes_distribution[1]); //duplica el numero de instancias positivas
	
	double rand = Math.random();
	
	int integerPart = (int)Math.floor(factor);
	
	double decimalPart = factor - integerPart;
	
	if (rand < decimalPart)
		replication = integerPart + 1;
	else
		replication = integerPart;
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
    
    //System.out.println(value.toString());
    //System.out.println("Label: "+label);
    
	LongWritable id;
	Random r = new Random();
	
	if (!noOutput) {  		
		if(value.toString().endsWith(","+negativeClass)){
    		int random = r.nextInt(replication); 
    		id = new LongWritable(random);		
    		context.write(id, value);  
		}
		else{
			for(int i = 0 ; i < replication ; i++){   
				id = new LongWritable(i);
				context.write(id, value);  
			}
		}	
	}    	  
  }

  /*
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
  
    Instance instance = converter.convert(value.toString());
    
    int label = (int)dataset.getLabel(instance);
	
	double factor = (classes_distribution[negativeClass] / classes_distribution[label]);
	
	double rand = Math.random();
	
	int integerPart = (int)Math.floor(factor);
	
	double decimalPart = factor - integerPart;
	
	int replication;
	
	if (rand < decimalPart)
		replication = integerPart + 1;
	else
		replication = integerPart;
	
	LongWritable id = new LongWritable(1);
	
	if (!noOutput) {  
      for(int i = 0 ; i < replication ; i++){   
	    context.write(id, value);  
      }
	}
    	  
  }*/
  
  
  

}
