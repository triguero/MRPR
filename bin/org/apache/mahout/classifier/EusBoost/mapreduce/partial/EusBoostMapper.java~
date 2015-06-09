package org.apache.mahout.classifier.smo.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.smo.mapreduce.MapredOutput;
import org.apache.mahout.classifier.smo.mapreduce.Builder;
import org.apache.mahout.classifier.smo.mapreduce.MapredMapper;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.DataConverter;
import org.apache.mahout.classifier.basic.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SMOMapper extends MapredMapper<LongWritable,Text,StrataID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(SMOMapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  public int getFirstTreeId() {
    return firstId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    context.progress();
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), Builder.getHeader(conf), Builder.getTestPath(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
   */
  protected void configure(int partition, int numMapTasks, String header, String testPath) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    this.header=header;
    this.testPath=new Path(testPath);

    log.debug("partition : {}", partition);
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));
    context.progress();
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // prepare the data
    log.debug("partition: {} numInstances: {}", partition, instances.size());
        
    context.progress();

    Data data = new Data(getDataset(), instances);
    
    log.info("SMO: Size of data partition= "+data.size());
    
   // log.info("cabecera: "+header);
    
    context.progress();

    Utils.readHeader(this.header);
    
    // Leer el conjunto de test.
    
    
    //System.out.println("Test set: "+testPath);
    
    // cargar los datos.
    Configuration conf = context.getConfiguration();

    FileSystem hfs2 = testPath.getFileSystem(conf);
   // PrototypeSet testSet= SMOUtils.readTest(hfs2, this.testPath);   
     
    
    log.info("Partition "+partition);
    
    
    try {
		//smo_algorithm.build(data, testSet, header, context);
    	smo_algorithm.build(data, hfs2, testPath, header, context,partition);
	} catch (Exception e) {
		// TODO Bloque catch generado automáticamente
		e.printStackTrace();
	}
    
    context.progress();

    
    ArrayList<Integer> Pred= smo_algorithm.getPredictions();
    
    
    StrataID key = new StrataID();


    
    key.set(partition, firstId + 1);
    
    /*
    System.out.println("Pred size= "+Pred.length);
    System.out.println("Clases = "+data.getDataset().nblabels());
	  
	  for(int i=0; i<Pred.length;i++){
		  System.out.print(Pred[i]+",");
	  }
	  */
    // write predictions and number of labels
    MapredOutput emOut = new MapredOutput(Pred, data.getDataset().nblabels());
      
    // emOut.getRS().print();
    context.write(key, emOut);

   
  }
}
