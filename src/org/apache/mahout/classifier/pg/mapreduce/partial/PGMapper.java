package org.apache.mahout.classifier.pg.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.pg.mapreduce.MapredOutput;
import org.apache.mahout.classifier.pg.mapreduce.Builder;
import org.apache.mahout.classifier.pg.mapreduce.MapredMapper;
import org.apache.mahout.classifier.pg.utils.PGUtils;
import org.apache.mahout.classifier.pg.data.Data;
import org.apache.mahout.classifier.pg.data.DataConverter;
import org.apache.mahout.classifier.pg.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.HashMap;

public class PGMapper extends MapredMapper<LongWritable,Text,StrataID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(PGMapper.class);
  
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
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), Builder.getHeader(conf), Builder.getWindows(conf));
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
  protected void configure(int partition, int numMapTasks, String header, int windows) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    this.header=header;
    this.windows=windows;
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
    

    log.info("PG: Size of data partition= "+data.size());
    
    log.info("cabecera: "+header);
    
    context.progress();

    PGUtils.readHeader(this.header);
    
    System.out.println("number of windows en map: "+this.windows);
    
    try {
		pg_algorithm.build(data, context, this.windows);
	} catch (Exception e) {
		// TODO Bloque catch generado automáticamente
		e.printStackTrace();
	}
    
    context.progress();

    
    PrototypeSet RS = pg_algorithm.reduceSet();
    
    
    // Prueba a escribir el dataset en cada map
    
    System.out.println("Guardo el dataset como ReducidoPG.data en ~: "+RS.size());
	FileSystem outFS;  
    Configuration conf = context.getConfiguration();
    Path outputPath = new Path("/user/root/");
	outFS = outputPath.getFileSystem(conf);
	FSDataOutputStream ofile = null;
	Path filenamePath = new Path(outputPath, "Reducido"+pg_algorithm.PGmethod).suffix(".data");
	
	if(!outFS.exists(filenamePath))
		ofile = outFS.create(filenamePath);
	else
		ofile = outFS.append(filenamePath);
	
			  
    //Check which is the data type of the inputs
    HashMap<Integer,Boolean> nominalInput = new HashMap<Integer, Boolean>();
    for (int i=0; i<RS.get(0).numberOfInputs(); i++)
        nominalInput.put(i, (Attributes.getInputAttribute(i).getType()==Attribute.NOMINAL));

    //Check which is the data type of the outputs
    boolean nominal_output = (Attributes.getOutputAttribute(0).getType()==Attribute.NOMINAL);
    
    for(Prototype p: RS)
    {
       // System.out.print("voy,");

        Prototype q = p.denormalize(); //TOKADO PARA NO NORMALIZAR
        for(int i=0; i<RS.get(0).numberOfInputs(); ++i)
        {
            if(nominalInput.get(i))
            {
				ofile.writeBytes(q.getInputAsNominal(i) + ",");

                //text += q.getInputAsNominal(i) + ", ";
            }
            else
            {
                double q_i = q.getInput(i); //
                // p.print();
                //System.out.println("q_i" + q_i);
                if(Prototype.getTypeOfAttribute(i) == Prototype.INTEGER)
                	ofile.writeBytes(Math.round(q_i) + ","); // ERROR de DIEGO!?
                else if(Prototype.getTypeOfAttribute(i) == Prototype.DOUBLE)
                	ofile.writeBytes(q_i + ",");  
            }
        }
       if(nominal_output)
    	   ofile.writeBytes(q.getOutputAsNominal(0) + "\n");
       else
    	   ofile.writeBytes(q.label() + "\n");
    }
   
ofile.close();

    
    
    
    
    StrataID key = new StrataID();

    key.set(partition, firstId + 1);
      
    //if (!isNoOutput()) {
    MapredOutput emOut = new MapredOutput(RS);
      
    // emOut.getRS().print();
    context.write(key, emOut);
    //}
   
  }
}
