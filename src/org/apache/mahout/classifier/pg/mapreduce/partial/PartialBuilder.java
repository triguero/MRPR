package org.apache.mahout.classifier.pg.mapreduce.partial;

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
import org.apache.mahout.classifier.pg.*;
import org.apache.mahout.classifier.pg.builder.PGgenerator;
import org.apache.mahout.classifier.pg.mapreduce.*;
import org.apache.mahout.classifier.pg.utils.PGUtils;
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

  public PartialBuilder(PGgenerator pg_algorithm, Path dataPath, Path datasetPath, String reduceType, String cabecera, int windows) {
    this(pg_algorithm, dataPath, datasetPath, reduceType, new Configuration(), cabecera,windows);
  }
  
  public PartialBuilder(PGgenerator pg_algorithm,
                        Path dataPath,
                        Path datasetPath, String reduceType,
                        Configuration conf, String cabecera, int windows) {
    super(pg_algorithm, dataPath, datasetPath, reduceType, conf, cabecera,windows);
  }

  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    job.setOutputKeyClass(StrataID.class);
    job.setOutputValueClass(MapredOutput.class);
    
    job.setMapperClass(PGMapper.class);
    
    // Elegir el reducer adecuado:
    
   
    if(this.reducePhase.equalsIgnoreCase("Join")){
        job.setReducerClass(JoinReducer.class); 
    }else if(this.reducePhase.equalsIgnoreCase("Fusion")){
        job.setReducerClass(FusionReducer.class); 
    }else if(this.reducePhase.equalsIgnoreCase("Filtering")){
        job.setReducerClass(FilteringReducer.class); 
    }else if(this.reducePhase.equalsIgnoreCase("IterativeFusion")){
    	job.setReducerClass(FusionIterativeReducer.class); 
    }else if(this.reducePhase.equalsIgnoreCase("FastFusion")){
    	job.setReducerClass(FastFusionReducer.class); 
    }else if(this.reducePhase.equalsIgnoreCase("IterativeFiltering")){
    	job.setReducerClass(FilteringIterativeReducer.class); 
    }else {
    	job.setNumReduceTasks(0); // no reducers
    }
    
   // job.setNumReduceTasks(10);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }
  
  
  
  @Override
  protected PrototypeSet parseOutput(Job job) throws IOException {
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
  
  
  protected PrototypeSet processOutput(JobContext job,
                                      Path outputPath) throws IOException { 
    Configuration conf = job.getConfiguration();

    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = PGUtils.listOutputFiles(fs, outputPath);

  //  System.out.println("Outfiles size= "+outfiles.length);
 
   //read the output (un solo fichero por ser reduce)
    MapredOutput value=null;
    for (Path path : outfiles) {
      for (Pair<StrataID,MapredOutput> record : new SequenceFileIterable<StrataID, MapredOutput>(path, conf)) {
        value = record.getSecond();
        
       // System.out.println("Size = "+ value.getRS().size());
        //return value.getRS();
      }
    }
    
    // cojo el último, porque es iterativo, ó el único que hay si 
    // lo hago todo con solo reduce.
    
	return value.getRS();
    

  }
  
  
}

