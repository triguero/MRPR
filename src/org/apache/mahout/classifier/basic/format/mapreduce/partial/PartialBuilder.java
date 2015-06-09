package org.apache.mahout.classifier.basic.format.mapreduce.partial;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.basic.*;
import org.apache.mahout.classifier.basic.format.mapreduce.*;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Dataset.InstanceSet;

import java.io.IOException;
import java.util.Arrays;

/**
 * Builds a model using partial data. Each mapper uses only the data given by its InputSplit
 */
public class PartialBuilder extends Builder {

  public PartialBuilder(Path dataPath, Path datasetPath, String cabecera) {
    this(dataPath, datasetPath,new Configuration(), cabecera);
  }
  
  public PartialBuilder(Path dataPath,
                        Path datasetPath, 
                        Configuration conf, String cabecera) {
    super(dataPath, datasetPath, conf, cabecera);
  }

  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    job.setOutputKeyClass(StrataID.class);
    job.setOutputValueClass(rangesOutput.class);
    
    job.setMapperClass(SetRangesMapper.class);
    
    // Elegir el reducer adecuado:
    

    job.setReducerClass(SetRangesReducer.class); 

   // job.setNumReduceTasks(10);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }
  
  
  
  @Override
  protected org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Pair<double[],double[]> parseOutput(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    // int numMaps = Builder.getNumMaps(conf);
    
    Path outputPath = getOutputPath(conf);
          
    
    return processOutput(job, outputPath);
  }
  
  
  /**
   * Processes the output from the output path.<br>
   * 
   * @param outputPath
   *          directory that contains the output of the job
   * @param keys
   *          can be null
   * @param trees
   *          can be null
   * @throws java.io.IOException
   */
  
  
  protected org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Pair<double[],double[]>  processOutput(JobContext job,
                                      Path outputPath) throws IOException { 
    Configuration conf = job.getConfiguration();

    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = Utils.listOutputFiles(fs, outputPath);

  //  System.out.println("Outfiles size= "+outfiles.length);
 
   //read the output (un solo fichero por ser reduce)
    rangesOutput value=null;
    for (Path path : outfiles) {
      for (Pair<StrataID,rangesOutput> record : new SequenceFileIterable<StrataID, rangesOutput>(path, conf)) {
        value = record.getSecond();
        
       // System.out.println("Size = "+ value.getRS().size());
        //return value.getRS();
      }
    }
    
    // cojo el último, porque es iterativo, ó el único que hay si 
    // lo hago todo con solo reduce.
    
	return value.getMaxMins();
    

  }
  
  
}

