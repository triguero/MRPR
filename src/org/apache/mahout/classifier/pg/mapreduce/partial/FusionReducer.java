package org.apache.mahout.classifier.pg.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.pg.mapreduce.MapredOutput;
import org.apache.mahout.classifier.pg.mapreduce.Builder;
import org.apache.mahout.classifier.pg.mapreduce.MapredReducer;
import org.apache.mahout.classifier.pg.utils.PGUtils;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.mahout.classifier.pg.data.Data;
import org.apache.mahout.classifier.pg.data.DataConverter;
import org.apache.mahout.classifier.pg.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ICPL.ICPLGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class FusionReducer extends MapredReducer<StrataID,MapredOutput,StrataID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(PGReducer.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
 
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
   

  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    log.info("Configuring reducer");
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), Builder.getHeader(conf));
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
  protected void configure(int partition, int numMapTasks, String header) {
   // converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.strata = partition;
    this.header=header;
    log.debug("partition : {}", partition);
  }
   
 // KEYIN id, Iterable<VALUEIN> rs, 
// aquí no funciona con los tipos de datos concretos, uso abstractos en MapredReducer
  /*
  protected void reduce(StrataID id, MapredOutput rs, Context context) throws IOException, InterruptedException {
	  log.info("añadiendo instancias "+rs.getRS().size());
	  join.add(rs.getRS());
	  
	//  context.write(id, new PrototypeSet(rs.getRS()));
	  //instances.add(converter.convert(value.toString()));
  }
  */

  // para hacerlo bien, esta fase hay que hacerla en el principal, no en el mapper.

  
  protected void cleanup(Context context) throws IOException, InterruptedException {
	 // log.debug("partition: {} numInstances: {}", partition, instances.size());
    
	    log.info("reduce: {} numInstances: {}", strata, join.size());
	   	   
	    StrataID key = new StrataID();

	    key.set(strata, firstId + 1);
	      
	    //join.print();
	    
	    // hacer fusión.
	    
	    log.info("cabecera: "+header);
	    
	    PGUtils.readHeader(this.header);
	    
	    context.progress();
	    join=new PrototypeSet(merge(join, context));
	    context.progress();
	    log.info("Tamaño despues de la fusión: {} ", join.size());

	    if (!isNoOutput()) {
	    	MapredOutput emOut = new MapredOutput(join);
	        context.write(key, emOut);
	    }
	    	    
	    //save it in disk
	   // join.save("prueba.dat");

  }
 
  
  public PrototypeSet merge(PrototypeSet initial, Context context){
	  
	  ICPLGenerator algorithm = new ICPLGenerator(initial,2,"RT2",1,0, context);
	  
	  return algorithm.reduceSet();
	  
	  /*
	  	  //PrototypeSet merge=new PrototypeSet();


	  for(int i=0; i<initial.size();i++){
		  
		  //initial.nearestTo(initial.get(i));
	  }
	  */
  }
  
}
