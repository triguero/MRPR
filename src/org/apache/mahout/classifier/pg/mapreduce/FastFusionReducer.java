package org.apache.mahout.classifier.pg.mapreduce;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.classifier.pg.builder.PGgenerator;
import org.apache.mahout.classifier.pg.data.Dataset;
import org.apache.mahout.classifier.pg.mapreduce.partial.StrataID;
import org.apache.mahout.classifier.pg.utils.PGUtils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.GMCA.GMCAGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.ICPL.ICPLGenerator;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This Mapred allows to run more than one reducers.
 * 
 */
public class FastFusionReducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Reducer<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected PGgenerator pg_algorithm;
  
  private Dataset dataset;
  protected String header;

  protected PrototypeSet join = new PrototypeSet();
  protected int strata;
  private int firstId = 0;

  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected PGgenerator getPGgeneratorBuilder() {
    return pg_algorithm;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getPGgeneratorBuilder(conf), Builder.loadDataset(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, PGgenerator pg_algorithm, Dataset dataset, String header) {
    Preconditions.checkArgument(pg_algorithm != null, "PGgenerator not found in the Job parameters");
    this.noOutput = noOutput;
    this.pg_algorithm = pg_algorithm;
    this.dataset = dataset;
    this.header = header;
  }

  
  /**
   * Generic reducer, it only adds all the RSs.
   */
  
protected void reduce(KEYIN id, Iterable<VALUEIN> rs, Context context)
		throws IOException, InterruptedException {
	// TODO Apéndice de método generado automáticamente
	
	//System.out.println("Si paso por aquí: "+id);
	//strata = (StrataID) id;

	for(VALUEIN value: rs){
		MapredOutput prueba = (MapredOutput) value;
		PrototypeSet strato = prueba.getRS();
		
	    PGUtils.readHeader(this.header);
	    
   		//System.out.println("apriori: "+join.size()+","+strato.size());
	    context.progress();
	    
   		PrototypeSet fusion=new PrototypeSet();
   		
   		
   		if(strato.size()>15)
   			fusion=new PrototypeSet(merge(strato, context));
	    
   		context.progress();

   		
   		// CAMBIOS PARA ESCRIBIR DATA DIRECTO A HDFS
	    if(fusion.size()>0)
	    	join.add(fusion);
	    else
	    	join.add(strato);
	       		
   		/*
    	System.out.println("Guardo el dataset como ReducidoPG.data en ~: "+join.size());
    	FileSystem outFS;  
        Configuration conf = context.getConfiguration();
        Path outputPath = new Path("/user/root/");
		outFS = outputPath.getFileSystem(conf);
		FSDataOutputStream ofile = null;
		Path filenamePath = new Path(outputPath, "ReducidoFastFusion").suffix(".data");
		
		if(!outFS.exists(filenamePath))
			ofile = outFS.create(filenamePath);
		else
			ofile = outFS.append(filenamePath);
		
				  
        //Check which is the data type of the inputs
        HashMap<Integer,Boolean> nominalInput = new HashMap<Integer, Boolean>();
        for (int i=0; i<join.get(0).numberOfInputs(); i++)
            nominalInput.put(i, (Attributes.getInputAttribute(i).getType()==Attribute.NOMINAL));

        //Check which is the data type of the outputs
        boolean nominal_output = (Attributes.getOutputAttribute(0).getType()==Attribute.NOMINAL);
        
        for(Prototype p: join)
        {
           // System.out.print("voy,");

            Prototype q = p.denormalize(); //TOKADO PARA NO NORMALIZAR
            for(int i=0; i<join.get(0).numberOfInputs(); ++i)
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
    */
   		
   		//System.out.println("Resultados: "+join.size()+","+strato.size()+","+fusion.size());
    	
		// if you write here, the cleanup does not work.
		//MapredOutput salida= new MapredOutput(join);
		//context.write((KEYOUT) id, (VALUEOUT) salida);
	}
	
	//System.out.println("*******************");


}

protected void cleanup(Context context) throws IOException, InterruptedException {
	 
    System.out.println("escribo el join.");
    StrataID key = new StrataID();

    key.set(strata, firstId + 1);
    
	MapredOutput salida= new MapredOutput(join);
	context.write((KEYOUT) key, (VALUEOUT) salida);
}


 
  
  public PrototypeSet merge(PrototypeSet initial, Context context){
	  
	//  ICPLGenerator algorithm = new ICPLGenerator(initial,2,"RT2",1,0, context);
	  GMCAGenerator algorithm = new GMCAGenerator(initial,context);
	  return algorithm.reduceSet();
	  
	  /*
	  	  //PrototypeSet merge=new PrototypeSet();


	  for(int i=0; i<initial.size();i++){
		  
		  //initial.nearestTo(initial.get(i));
	  }
	  */
  }

}


