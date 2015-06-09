package org.apache.mahout.classifier.smo.mapreduce;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.classifier.smo.builder.SMOgenerator;
import org.apache.mahout.classifier.basic.data.Data;
import org.apache.mahout.classifier.basic.data.DataLoader;
import org.apache.mahout.classifier.basic.data.Dataset;
import org.apache.mahout.classifier.smo.mapreduce.partial.PartialBuilder;
import org.apache.mahout.classifier.basic.utils.Utils;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.*;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Dataset.Attributes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

public class SMOModel extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(SMOModel.class);
  
  private Path dataPath;
  private Path datasetPath;
  private Path testPath;
  private Path headerPath;
  private Path outputPath;
  private Path timePath;
  
  private String dataName;
  private String testName;
  private String reducePhase="Majority";
  
  private long time;
  
 
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
	// Primero lectura de parámetros y control de que no falte:
	
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
        .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
        .withDescription("The path of the file descriptor of the dataset").create();

    Option testOpt = obuilder.withLongName("test").withShortName("tst").withRequired(true)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Test Data path").create();
    
    Option header = obuilder.withLongName("header").withShortName("he").withRequired(true)
            .withArgument(abuilder.withName("header").withMinimum(1).withMaximum(1).create())
            .withDescription("Header of the dataset in Keel format").create();
    
    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Output path, will contain the preprocessed dataset").create();
  
    Option reduceType = obuilder.withLongName("TypeOfReduce").withShortName("r").withRequired(false)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Type of reduce: Majority. Default: Majority").create();
    
    
    Option helpOpt = obuilder.withLongName("help").withShortName("h")
        .withDescription("Print out help").create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(testOpt).withOption(datasetOpt).withOption(header).withOption(outputOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }
      

      // Obtenemos los parámetros que nos interesen:
      
      dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      testName = cmdLine.getValue(testOpt).toString();
      String outputName = cmdLine.getValue(outputOpt).toString();
      String headerName = cmdLine.getValue(header).toString();

      if (cmdLine.hasOption(reduceType))
    	  this.reducePhase= cmdLine.getValue(reduceType).toString();

      if (log.isDebugEnabled()) {
        log.debug("data : {}", dataName);
        log.debug("dataset : {}", datasetName);
        log.debug("header : {}", header);
        log.debug("output : {}", outputName);
        log.debug("reduceType : {}", reducePhase);
      }

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      testPath = new Path(testName);
      outputPath = new Path(outputName);
      headerPath = new Path(headerName);

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    buildModel();
    
    return 0;
  }
  

  private void buildModel() throws IOException, ClassNotFoundException, InterruptedException {
  //	  make sure the output path does not exist
    FileSystem ofs = outputPath.getFileSystem(getConf());

    
    if (ofs.exists(outputPath)) {
      log.error("Output path already exists");
      System.out.println("Error");
      return;
    }

    SMOgenerator smo_algorithm = new SMOgenerator();
    // crear valores para los atributos en formato InstaceSEt y Prototypset
       
    
    FileSystem hfs = headerPath.getFileSystem(getConf());
    InstanceSet cabecera= Utils.readHeader(hfs, this.headerPath);
    
    // Here We refer to the mapper/reducer classes, establish the corresponding key and value classes.
    
    log.info("SMO: Partial Mapred implementation"); 
        
    Builder modelBuilder = new PartialBuilder(smo_algorithm, dataPath, datasetPath, testName, reducePhase, getConf(), cabecera.getHeader());
    
    modelBuilder.setOutputDirName(outputPath.toString());
 
    
    time = System.currentTimeMillis();
    ArrayList<Integer> resultingPredictions=modelBuilder.build();
    
    
    time = System.currentTimeMillis() - time;
    log.info("SMO: Build Time: {}", Utils.elapsedTime(time));
    log.info("SMO: Build Time in seconds: {}", Utils.elapsedSeconds(time));
   
    // store the building time in a file to post-process:
	FileSystem outFS = outputPath.getFileSystem(getConf());
	Path filenamePath = new Path(outputPath, "BuildingTime").suffix(".txt");
	FSDataOutputStream ofile = null;

    ofile = outFS.create(filenamePath);
	ofile.writeUTF("\n"+Utils.elapsedSeconds(time));
	ofile.close();		
	
	// Escribiendo predicciones.
	filenamePath = new Path(outputPath, "Predictions").suffix(".txt");
   
	ofile = outFS.create(filenamePath);
	
	for(int i=0; i<resultingPredictions.size();i++){
    	String dato = Integer.toString(resultingPredictions.get(i))+"\n";
 		ofile.writeBytes(dato);
	}
	ofile.close();	
	

   // SMOUtils.storeWritable(getConf(), modelPath, resultingSet);
   

    
  }
  
  protected static Data loadData(Configuration conf, Path dataPath, Dataset dataset) throws IOException {
    log.info("FS: Loading the data...");
    FileSystem fs = dataPath.getFileSystem(conf);
    Data data = DataLoader.loadData(dataset, fs, dataPath);
    log.info("FS: Data Loaded");
    
    return data;
  }
  
  private void writeToFileBuildTime(String time) throws IOException{	
    FileSystem outFS = timePath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;		
	Path filenamePath = new Path(timePath, dataName + "_build_time").suffix(".txt");
	try    
	  {	        	
        if (ofile == null) {
	      // this is the first value, it contains the name of the input file
	      ofile = outFS.create(filenamePath);
		  // write the Build Time	      	      	      	      
		  StringBuilder returnString = new StringBuilder(200);	      
	      returnString.append("=======================================================").append('\n');
		  returnString.append("Build Time\n");
		  returnString.append("-------------------------------------------------------").append('\n');
		  returnString.append(
		  StringUtils.rightPad(time,5)).append('\n');                  
		  returnString.append("-------------------------------------------------------").append('\n');	      				
		  String output = returnString.toString();
	      ofile.writeUTF(output);
		  ofile.close();		  
		} 	    
      } 
	finally 
      {
	    Closeables.closeQuietly(ofile);
	  }
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new SMOModel(), args);
  }
  
}
