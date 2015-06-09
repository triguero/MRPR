package org.apache.mahout.classifier.df.mapreduce;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.mapreduce.resampling.UndersamplingMapper_v2;
import org.apache.mahout.classifier.df.mapreduce.resampling.UndersamplingReducer_v2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UndersamplingBuilder_v2 {

	  private static final Logger log = LoggerFactory.getLogger(UndersamplingBuilder.class);
	  private final Path dataPath;
	  private final Path datasetPath;
	  private final Configuration conf;
	  private final Path outputPath;
	  private String outputDirName = "output_preprocessing";
	  
	  protected UndersamplingBuilder_v2(Path dataPreprocessingPath, Path dataPath, Path datasetPath, Configuration conf) {	    
	    this.dataPath = dataPreprocessingPath;
		this.datasetPath = datasetPath;	    
		this.conf = new Configuration(conf);
		this.outputPath = dataPath;
	  }
	  
	  public static int getNumMaps(Configuration conf) {
	    return conf.getInt("mapred.map.tasks", -1);
	  }
	  
	  public static Path getDistributedCacheFile(Configuration conf, int index) throws IOException {
	    URI[] files = DistributedCache.getCacheFiles(conf);
		    
	    if (files == null || files.length <= index) {
	      throw new IOException("path not found in the DistributedCache");
	    }
		    
	    return new Path(files[index].getPath());
	  }
	  
	  public static Dataset loadDataset(Configuration conf) throws IOException {
	    Path datasetPath = getDistributedCacheFile(conf, 0);
		    
		return Dataset.load(conf, datasetPath);
	  }
	  
	  protected Path getOutputPath(Configuration conf) throws IOException {
	    // the output directory is accessed only by this class, so use the default
		// file system
		FileSystem fs = FileSystem.get(conf);
		return new Path(fs.getWorkingDirectory(), outputDirName);
	  }
	  
	  public static boolean isOutput(Configuration conf) {
	    return conf.getBoolean("debug.mahout.preprocessing.output", true);
	  }
	  
	  protected void configureJob(Job job) throws IOException {
	    Configuration conf = job.getConfiguration();	    
	    job.setJarByClass(UndersamplingBuilder.class); 
	  
	    FileInputFormat.addInputPath(job, dataPath);
	    FileOutputFormat.setOutputPath(job, outputPath);
	  
	    job.setMapperClass(UndersamplingMapper_v2.class);	 
	    job.setReducerClass(UndersamplingReducer_v2.class);
	  
	    job.setOutputKeyClass(NullWritable.class);
	    job.setOutputValueClass(Text.class);	
	    
	    job.setMapOutputKeyClass(LongWritable.class);
	    job.setMapOutputValueClass(Text.class);
	  }
	  
	  protected boolean runJob(Job job) throws ClassNotFoundException, IOException, InterruptedException {
	    return job.waitForCompletion(true);
	  }
	  
	  public void build() throws IOException, ClassNotFoundException, InterruptedException {
	    Path outputPath = getOutputPath(conf);
		FileSystem fs = outputPath.getFileSystem(conf);
		    
	    // check the output
		if (fs.exists(outputPath)) {
		  throw new IOException("Output path already exists : " + outputPath);
		}
	    // put the dataset into the DistributedCache
		DistributedCache.addCacheFile(datasetPath.toUri(), conf);
		    
		Job job = new Job(conf, "undersampling builder");
		    
	    log.debug("Configuring the job...");
	    configureJob(job);
		    
		log.debug("Running the job...");
		if (!runJob(job)) {
	      log.error("Job failed!");	
		}  
	  }
	  
	  /**
	   * sort the splits into order based on size, so that the biggest go first.<br>
	   * This is the same code used by Hadoop's JobClient.
	   * 
	   * @param splits
	   *          input splits
	   */
	  public static void sortSplits(InputSplit[] splits) {
	    Arrays.sort(splits, new Comparator<InputSplit>() {
	      @Override
	      public int compare(InputSplit a, InputSplit b) {
	        try {
	          long left = a.getLength();
	          long right = b.getLength();
	          if (left == right) {
	            return 0;
	          } else if (left < right) {
	            return 1;
	          } else {
	            return -1;
	          }
	        } catch (IOException ie) {
	          throw new IllegalStateException("Problem getting input split size", ie);
	        } catch (InterruptedException ie) {
	          throw new IllegalStateException("Problem getting input split size", ie);
	        }
	      }
	    });
	  }

	}
