package org.apache.mahout.classifier.feature_selection.mapreduce;

import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.DataConverter;
import org.apache.mahout.classifier.basic.data.DataLoader;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.basic.data.Instance;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mapreduce implementation that classifies with KNN the Input data using a previously preprocessed data
 */

public class FeatureSeparator {
	
  private static final Logger log = LoggerFactory.getLogger(FeatureSeparator.class);
  private final Path preprocessedPath;
  private final Path inputPath;
   private final Path datasetPath;

  private final Configuration conf;
  private String cabecera;

  private final Path outputPath; // path that will containt the final output of the classifier
  private final Path mappersOutputPath; // mappers will output here
  private double[][] results;
	  
  private int numSelectedFeatures=0;
  
  public double[][] getResults() {
    return results;
  }
  
  // orden: dataset, info, RS, salida
  
  public FeatureSeparator(Path inputPath, Path datasetPath, Path preprocessedPath, Path outputPath, Configuration conf, String cabecera) {
    this.inputPath = inputPath;
    this.preprocessedPath = preprocessedPath;
    this.datasetPath = datasetPath;
    this.outputPath = outputPath;
    this.conf = conf;
    this.cabecera=cabecera;

    mappersOutputPath = new Path(outputPath, "mappers");
  }
  
  
  public FeatureSeparator(Path inputPath, Path datasetPath, Path preprocessedPath, Path outputPath, Configuration conf, String cabecera, int numSelectedFeatures) {
	    this.inputPath = inputPath;
	    this.preprocessedPath = preprocessedPath;
	    this.datasetPath = datasetPath;
	    this.outputPath = outputPath;
	    this.conf = conf;
	    this.cabecera=cabecera;
	    this.numSelectedFeatures = numSelectedFeatures;
	    mappersOutputPath = new Path(outputPath, "mappers");
}
  
  
  private void configureJob(Job job) throws IOException {
    job.setJarByClass(FeatureSeparator.class);

	FileInputFormat.setInputPaths(job, inputPath);
	FileOutputFormat.setOutputPath(job, mappersOutputPath);

	job.setOutputKeyClass(DoubleWritable.class);
	job.setOutputValueClass(Text.class);

	job.setMapperClass(ClassifierMapper.class);
	job.setNumReduceTasks(0); // no reducers

	job.setInputFormatClass(ClassifierTextInputFormat.class);
	job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }

  /**
   * Mandatory to send the header to the mappers.
   * 
   * @param conf
   * @param header
   */
  private static void setHeader(Configuration conf, String header) {
	    conf.set("mahout.fc.InstanceSet", StringUtils.toString(header));
  }
  
  public static String getHeader(Configuration conf){
	    String string = conf.get("mahout.fc.InstanceSet");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
  
  /**
   * Mandatory to send the 
   * 
   * @param conf
   * @param header
   */
  private static void setNumSelectedFeatures(Configuration conf, Integer num) {
	    conf.set("mahout.fc.Integer", StringUtils.toString(num));
  }
  
  public static Integer getNumSelectedFeatures(Configuration conf){
	    String string = conf.get("mahout.fc.Integer");
	    if (string == null) {
	      return null;
	    }
	    
	    return StringUtils.fromString(string); 
  }
  
  
  public String getNewHeader(){
	  return this.cabecera;
  }
  
  public void run() throws IOException, ClassNotFoundException, InterruptedException {
    FileSystem fs = FileSystem.get(conf);

	// check the output
	if (fs.exists(outputPath)) {
	  throw new IOException("FS: Output path already exists : " + outputPath);
	}

	
    setHeader(conf, cabecera);

    setNumSelectedFeatures(conf,this.numSelectedFeatures);
    
    
	log.info("FS: Adding the dataset to the DistributedCache");
	// put the test set into the DistributedCache
	
	DistributedCache.addCacheFile(datasetPath.toUri(), conf);
	
	log.info("FS: Adding the preprocessed dataset to the DistributedCache");
	DistributedCache.addCacheFile(preprocessedPath.toUri(), conf);

	Job job = new Job(conf, "FS constructor: "+datasetPath.getName()+", "+this.inputPath.getName());

	log.info("FS: Configuring the job...");
	configureJob(job);

	log.info("FS: Running the job...");
	if (!job.waitForCompletion(true)) {
	  throw new IllegalStateException("FS: Job failed!");
	}

	parseOutput(job);

	HadoopUtil.delete(conf, mappersOutputPath);
  }
  
  /**
   * Extract the prediction for each mapper and write them in the corresponding output file. 
   * The name of the output file is based on the name of the corresponding input file.
   * Will compute the ConfusionMatrix if necessary.
   */
  private void parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    FileSystem fs = mappersOutputPath.getFileSystem(conf);

    Path[] outfiles = Utils.listOutputFiles(fs, mappersOutputPath);

    // read all the output
 //   List<double[]> resList = new ArrayList<double[]>();
    for (Path path : outfiles) {
      FSDataOutputStream ofile = null;
      FSDataOutputStream ofile2 = null;
      try {
        for (Pair<DoubleWritable,Text> record : new SequenceFileIterable<DoubleWritable,Text>(path, true, conf)) {
          double key = record.getFirst().get();
          String value = record.getSecond().toString();
          if (ofile == null) {
            // this is the first value, it contains the name of the input file
            ofile = fs.create(new Path(outputPath, value).suffix(".out"));
            ofile2 = fs.create(new Path(outputPath, value).suffix(".header"));
          } else {
            // The key contains the correct label of the data. The value contains a prediction
            
        	  if(value.contains("@"))
        		  ofile2.writeBytes(value);
        	  else{	  
        		  ofile.writeBytes(value); // write the instance
        	      ofile.writeBytes("\n");
        	  }

            //resList.add(new double[]{key, Double.valueOf(value)});
          }
        }
      } finally {
        Closeables.closeQuietly(ofile);
      }
    }
    
        
 //   System.out.println(cabecera);
    //results = new double[resList.size()][2];
    //resList.toArray(results);
  }
  
  /**
   * TextInputFormat that does not split the input files. This ensures that each input file is processed by one single
   * mapper.
   */
  private static class ClassifierTextInputFormat extends TextInputFormat {
    @Override
    protected boolean isSplitable(JobContext jobContext, Path path) {
      return false;
    //	 return true; // SI ESTÄ A FALSE NO TE DEJA HACER MÄS DE UN MAPPER!!!!!
    }
  }
  
  public static class ClassifierMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {

    /** used to convert input values to data instances */
    private DataConverter converter;
    private  boolean []features;
   // private final Random rng = RandomUtils.getRandom();
    private boolean first = true;
    private final Text lvalue = new Text();
    private Dataset test, preprocessed;
    
    protected String header;

    protected int numSelectedFeatures;

    
    
    private final DoubleWritable lkey = new DoubleWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);    //To change body of overridden methods use File | Settings | File Templates.

      Configuration conf = context.getConfiguration();
      
      header=FeatureSeparator.getHeader(conf);
      
      numSelectedFeatures=FeatureSeparator.getNumSelectedFeatures(conf);
      
    //  log.info("cabecera: "+header);
      Utils.readHeader(this.header);
      
      URI[] files = DistributedCache.getCacheFiles(conf);

      if (files == null || files.length < 2) {
        throw new IOException("not enough paths in the DistributedCache");
      }
       
      test = Dataset.load(conf, new Path(files[0].getPath()));

      context.progress();
      converter = new DataConverter(test);

      context.progress();
      System.out.println(files[1].getPath());
      
      if(numSelectedFeatures==0)
    	  features= MapredOutput.load(conf, new Path(files[1].getPath()),test.nbAttributes()-1);
      else
    	  features= MapredOutput.load2(conf, new Path(files[1].getPath()),test.nbAttributes()-1, numSelectedFeatures);


      
      // Limpiar la cabecera de keel.
	  String trozos[]= header.split("@");

	  	String newHeader="";
	  	int attribute=0;
	  	for(int i=0; i< trozos.length;i++){
	  	//	System.out.println(trozos[i]);
			
	  		if(trozos[i].contains("attribute") && !trozos[i].contains("class")){
	      		if(features[attribute])
	      			newHeader+="@"+trozos[i]+"\n";
	      		
	  			attribute++;
	  		}else{
      			newHeader+="@"+trozos[i]+"\n";

	  		}
	  	}
  	
	  	header=newHeader;
      context.progress();

      
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      if (first) {
        FileSplit split = (FileSplit) context.getInputSplit();
        Path path = split.getPath(); // current split path
        lvalue.set(path.getName());
        lkey.set(key.get());
        context.write(lkey, lvalue);

        first = false;
       
      }

      
      String line = value.toString();
      
	  String trozos[]= line.split(",");
	   
	  String cat="";
	  for(int i=0; i< trozos.length-1;i++){
		if(features[i]){
			  cat+=trozos[i]+",";
		 }
	  }
	  cat+=trozos[trozos.length-1];
      System.out.println(cat);
      
      lkey.set(1);
      lvalue.set(cat);
      context.write(lkey, lvalue);
      
     }
    
    protected void cleanup(Context context) throws IOException, InterruptedException {
        lkey.set(1);
        lvalue.set(header);
        context.write(lkey, lvalue);
        
    }
  }
}

