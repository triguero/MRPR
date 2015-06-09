//
//  Main.java
//
// Isaac Triguero
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.DE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithmHadoop;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.*;

/**
 * DE algorithm calling.
 * @author Isaac Triguero
 */
public class DEAlgorithmHadoop extends PrototypeGenerationAlgorithmHadoop<DEGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected DEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new DEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes DE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[]) throws Exception
    {
        DEAlgorithmHadoop isaak = new DEAlgorithmHadoop();
        
        if(args.length!=1){
        	System.err.println("Uso: DEAlgorithmHadoop <configuration file>");
        	System.exit(-1);
        }
        
        Parameters.assertBasicArgs(args);
	    readParametersFile(args[0]);
	    printParameters();
        
        Job job = new Job();
	    job.setJarByClass(DEAlgorithmHadoop.class);
	    job.setJobName("DEAlgorithmHadoop");
	    
	        
	    FileInputFormat.addInputPath(job, new Path(args[0]));
	    
	    job.setMapperClass(DEMapper.class);
	    job.setReducerClass(DEReducer.class);
	    
	    /*
	    
	    
	    job.setMapOutputKeyClass(theClass);
	    job.setMapOutputValueClass(theClass);
	    job.setOutputKeyClass(theClass);
	    job.setOutputValueClass();
        */
	    
	    
      /*  Job job = new Job(conf, "decision forest classifier");

        //log.info("Configuring the job...");
        isaak.configureJob(job);
        isaak.execute(args);
        
        */
        
    }


}
