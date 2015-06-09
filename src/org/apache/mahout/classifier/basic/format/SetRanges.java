package org.apache.mahout.classifier.basic.format;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
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
import org.apache.hadoop.mapreduce.Reducer;
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
import org.apache.mahout.classifier.basic.format.mapreduce.rangesOutput;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.classifier.pg.mapreduce.Builder;
import org.apache.mahout.classifier.pg.mapreduce.MapredOutput;
import org.apache.mahout.classifier.pg.mapreduce.MapredReducer;
import org.apache.mahout.classifier.pg.mapreduce.partial.PGReducer;
import org.apache.mahout.classifier.pg.mapreduce.partial.StrataID;
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
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Mapreduce implementation that classifies with KNN the Input data using a previously preprocessed data
 */

public class SetRanges {
	
  private static final Logger log = LoggerFactory.getLogger(SetRanges.class);
  private final Path inputPath;
   private final Path datasetPath;

  private final Configuration conf;
  private final String cabecera;
  private static int contador=0;

  private final Path outputPath; // path that will containt the final output of the classifier
  private final Path mappersOutputPath; // mappers will output here
  private double[][] results;
	  
  public double[][] getResults() {
    return results;
  }
  
  // orden: dataset, info, RS, salida
  
  public SetRanges(Path inputPath, Path datasetPath,  Path outputPath, Configuration conf, String cabecera) {
    this.inputPath = inputPath;
    this.datasetPath = datasetPath;
    this.outputPath = outputPath;
    this.conf = conf;
    this.cabecera=cabecera;

    mappersOutputPath = new Path(outputPath, "mappers");
  }
  
  private void configureJob(Job job) throws IOException {
    job.setJarByClass(SetRanges.class);

	FileInputFormat.setInputPaths(job, inputPath);
	FileOutputFormat.setOutputPath(job, mappersOutputPath);

	job.setOutputKeyClass(DoubleWritable.class);
	job.setOutputValueClass(Text.class);

	job.setMapperClass(SetRangesMapper.class);
	job.setReducerClass(SetRangesReducer.class); // no reducers

   
	job.setInputFormatClass(ClassifierTextInputFormat.class);
	job.setOutputValueClass(rangesOutput.class);
	
	//job.setInputFormatClass(ClassifierTextInputFormat.class);
	//job.setOutputFormatClass(SequenceFileOutputFormat.class);
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
  
  public void run() throws IOException, ClassNotFoundException, InterruptedException {
    FileSystem fs = FileSystem.get(conf);

	// check the output
	if (fs.exists(outputPath)) {
	  throw new IOException("KNN: Output path already exists : " + outputPath);
	}

	
    setHeader(conf, cabecera);

    
	log.info("SetRanges: Adding the dataset to the DistributedCache");
	// put the test set into the DistributedCache
	
	DistributedCache.addCacheFile(datasetPath.toUri(), conf);
	
	Job job = new Job(conf, "SetRanges: "+datasetPath.getName()+", "+this.inputPath.getName());

	log.info("SetRanges: Configuring the job...");
	configureJob(job);

	log.info("SetRanges: Running the job...");
	if (!job.waitForCompletion(true)) {
	  throw new IllegalStateException("SetRanges: Job failed!");
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
    List<double[]> resList = new ArrayList<double[]>();
    for (Path path : outfiles) {
      FSDataOutputStream ofile = null;
      try {
        for (Pair<DoubleWritable,Text> record : new SequenceFileIterable<DoubleWritable,Text>(path, true, conf)) {
          double key = record.getFirst().get();
          String value = record.getSecond().toString();
          if (ofile == null) {
            // this is the first value, it contains the name of the input file
            ofile = fs.create(new Path(outputPath, value).suffix(".out"));
          } else {
            // The key contains the correct label of the data. The value contains a prediction
            ofile.writeChars(value); // write the prediction
            ofile.writeChar('\n');

            resList.add(new double[]{key, Double.valueOf(value)});
          }
        }
      } finally {
        Closeables.closeQuietly(ofile);
      }
    }
    results = new double[resList.size()][2];
    resList.toArray(results);
  }
  
  /**
   * TextInputFormat that does not split the input files. This ensures that each input file is processed by one single
   * mapper.
   */
  private static class ClassifierTextInputFormat extends TextInputFormat {
    @Override
    protected boolean isSplitable(JobContext jobContext, Path path) {
     // return false;
    	 return true; // SI ESTÄ A FALSE NO TE DEJA HACER MÄS DE UN MAPPER!!!!!
    }
  }
  
  public static class SetRangesMapper extends Mapper<LongWritable, rangesOutput, DoubleWritable, rangesOutput> {

    /** used to convert input values to data instances */
    private DataConverter converter;
   // private final Random rng = RandomUtils.getRandom();
    private boolean first = true;
    private final Text lvalue = new Text();
    private Dataset test, preprocessed;
    
    protected String header;

    private double[] mins;
    private double[] maxs;
    private final DoubleWritable lkey = new DoubleWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);    //To change body of overridden methods use File | Settings | File Templates.

      Configuration conf = context.getConfiguration();
      
      header=SetRanges.getHeader(conf);
      
    //  log.info("cabecera: "+header);
      Utils.readHeader(this.header);
      
      URI[] files = DistributedCache.getCacheFiles(conf);

      if (files == null || files.length < 1) {
        throw new IOException("not enough paths in the DistributedCache");
      }
       
      test = Dataset.load(conf, new Path(files[0].getPath()));

      context.progress();
      converter = new DataConverter(test);

      context.progress();


      
    }

    @Override
    protected void map(LongWritable key, rangesOutput value, Context context) throws IOException, InterruptedException {
      if (first) {
        FileSplit split = (FileSplit) context.getInputSplit();
        Path path = split.getPath(); // current split path
        lvalue.set(path.getName());
        lkey.set(key.get());
       //context.write(lkey, lvalue);

        first = false;
        
        String line = value.toString();
        if (!line.isEmpty()) {
          Instance instance = converter.convert(line);
          mins=new double[instance.get().length-1];
          Arrays.fill(mins, Double.MAX_VALUE);
          maxs=new double[instance.get().length-1];
          Arrays.fill(maxs, Double.MIN_VALUE);
        }

      }
     // contador++;


      String line = value.toString();
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);
        
        for(int i=0; i<(instance.get().length-1);i++){
        	if(instance.get()[i]<mins[i]){
        		mins[i]=instance.get()[i];
      	    }
        	if(instance.get()[i]>maxs[i]){
        		maxs[i]=instance.get()[i];
        	}
        }
        
        lkey.set(test.getLabel(instance));
        lvalue.set(Double.toString(0));
        
        rangesOutput emOut = new rangesOutput(mins, maxs);

        
        context.write(lkey, emOut);
        
      }
      

    }
    
    
    protected void cleanup(Context context) throws IOException, InterruptedException {
    	
    	/*
    	System.out.println("COntador: "+contador);
    	
 	   String trozos[]= header.split("@");

  	   System.out.println("Trozos SIZE: "+trozos.length);
 	   
    	String newHeader="";
    	int attribute=0;
    	for(int i=0; i< trozos.length;i++){
    	//	System.out.println(trozos[i]);

    		if(trozos[i].contains("real") || trozos[i].contains("integer")  ){
    			newHeader+= "@"+trozos[i]+ "["+mins[attribute]+","+maxs[attribute]+"]\n";
    		}else{
    			newHeader+="@"+trozos[i]+"\n";
    		}
    		
    		if(trozos[i].contains("attribute") && !trozos[i].contains("class")){
    			attribute++;
    		}
    	}
    	*/
    	//System.out.println(newHeader);
    	//log.info(newHeader);

    }
  }
  
  public class SetRangesReducer extends Reducer<DoubleWritable,rangesOutput,DoubleWritable,rangesOutput> {
	  
	//  private static final Logger log = LoggerFactory.getLogger(SetRangesReducer.class);
	  
	  /** used to convert input values to data instances */
	  private DataConverter converter;
	  
	  /**first id */
	  private int firstId = 0;
	  
	    private double[] minsR;
	    private double[] maxsR;
	    
	    String header;
	 
	  /** will contain all instances if this mapper's split */
	  private final List<Instance> instances = Lists.newArrayList();
	   

	  
	  @Override
	  protected void setup(Context context) throws IOException, InterruptedException {
	    super.setup(context);
	    Configuration conf = context.getConfiguration();
	    log.info("Configuring reducer");
	    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf),SetRanges.getHeader(conf));
	  }
	  

	  protected void configure(int partition, int numMapTasks, String header) {
	   // converter = new DataConverter(getDataset());

		  this.header=header;
	    // mapper's partition
	  }

	  
	  protected void reduce(DoubleWritable id, Iterable<rangesOutput> rs, Context context)
				throws IOException, InterruptedException {
			// TODO Apéndice de método generado automáticamente
			
			System.out.println("Si paso por aquí: "+id);

			for(rangesOutput value: rs){
				context.progress();
				rangesOutput prueba = (rangesOutput) value;
			
				
		        if (minsR==null) {
		            minsR=new double[prueba.mins.length];
		            Arrays.fill(minsR, Double.MAX_VALUE);
		            maxsR=new double[prueba.mins.length];
		            Arrays.fill(maxsR, Double.MIN_VALUE);
		         }
		        
		        for(int i=0; i<prueba.mins.length;i++){
		        	if(prueba.mins[i]<prueba.mins[i]){
		        		minsR[i]=prueba.mins[i];
		      	    }
		        	if(prueba.maxs[i]>prueba.maxs[i]){
		        		maxsR[i]=prueba.maxs[i];
		        	}
		        }

			}
			

		}
	  
	  protected void cleanup(Context context) throws IOException, InterruptedException {
		 // log.debug("partition: {} numInstances: {}", partition, instances.size());
	    
	 	   String trozos[]= header.split("@");

	  	   System.out.println("Trozos SIZE: "+trozos.length);
	 	   
	    	String newHeader="";
	    	int attribute=0;
	    	for(int i=0; i< trozos.length;i++){
	    	//	System.out.println(trozos[i]);

	    		if(trozos[i].contains("real") || trozos[i].contains("integer")  ){
	    			newHeader+= "@"+trozos[i]+ "["+minsR[attribute]+","+maxsR[attribute]+"]\n";
	    		}else{
	    			newHeader+="@"+trozos[i]+"\n";
	    		}
	    		
	    		if(trozos[i].contains("attribute") && !trozos[i].contains("class")){
	    			attribute++;
	    		}
	    	}

	  }
  } // end reducer
	  

}
