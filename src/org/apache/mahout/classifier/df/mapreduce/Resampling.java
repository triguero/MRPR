package org.apache.mahout.classifier.df.mapreduce;

import java.io.IOException;

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
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

public class Resampling extends Configured implements Tool{

  private static final Logger log = LoggerFactory.getLogger(Resampling.class);
  
  private Path dataPath;
  
  private Path dataPreprocessingPath;
  
  private Path datasetPath;
  
  private Path timePath;
  
  private String dataName;
  
  private String dataPreprocessing;
  
  private String timeName;
  
  private long preprocessingTime;
  
  private boolean withOversampling = false; // use Oversampling technique
  
  private boolean withUndersampling = false; // use Undersampling technique
  
  private boolean withSmote = false; // use SMOTE technique
  
  private boolean withHybrid = false; // use Hybrid technique
  
  private boolean preprocessingTimeIsStored = false;
  
  private int partitions;
  
  private int npos; // number of instances of the positive class
  
  private int nneg; // number of instances of the negative class
  
  private int negclass; // name of the negative class
  
  public int run(String[] args) throws Exception, ClassNotFoundException, InterruptedException  {
	DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
	ArgumentBuilder abuilder = new ArgumentBuilder();
	GroupBuilder gbuilder = new GroupBuilder();
	
	Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
	        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
	        .withDescription("Data path").create();
	    
    Option dataPreprocessingOpt = obuilder.withLongName("dataPreprocessing").withShortName("dp").withRequired(true)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Data Preprocessing path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
        .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
        .withDescription("Dataset path").create();
    
    Option timeOpt = obuilder.withLongName("time").withShortName("tm").withRequired(false)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Time path").create();
	
    Option helpOpt = obuilder.withLongName("help").withShortName("h")
            .withDescription("Print out help").create();
    
    Option resamplingOpt = obuilder.withLongName("resampling").withShortName("rs").withRequired(true)
    		.withArgument(abuilder.withName("resampling").withMinimum(1).withMaximum(1).create())
    				.withDescription("The resampling technique (oversampling (overs), undersampling (unders) or SMOTE (smote))").create();
    
    Option nbpartitionsOpt = obuilder.withLongName("nbpartitions").withShortName("p").withRequired(true)
            .withArgument(abuilder.withName("nbpartitions").withMinimum(1).withMaximum(1).create())
            .withDescription("Number of partitions").create();
    
    Option nposOpt = obuilder.withLongName("npos").withShortName("npos").withRequired(true)
            .withArgument(abuilder.withName("npos").withMinimum(1).withMaximum(1).create())
            .withDescription("Number of instances of the positive class").create();
    
    Option nnegOpt = obuilder.withLongName("nneg").withShortName("nneg").withRequired(true)
            .withArgument(abuilder.withName("nneg").withMinimum(1).withMaximum(1).create())
            .withDescription("Number of instances of the negative class").create();
    
    Option negclassOpt = obuilder.withLongName("negclass").withShortName("negclass").withRequired(true)
            .withArgument(abuilder.withName("negclass").withMinimum(1).withMaximum(1).create())
            .withDescription("Name of the negative class").create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt).withOption(timeOpt)
        .withOption(helpOpt).withOption(resamplingOpt).withOption(dataPreprocessingOpt).withOption(nbpartitionsOpt).withOption(nposOpt)
        .withOption(nnegOpt).withOption(negclassOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
    
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }
      
      dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      dataPreprocessing = cmdLine.getValue(dataPreprocessingOpt).toString();
      String resampling = cmdLine.getValue(resamplingOpt).toString();
      partitions = Integer.parseInt(cmdLine.getValue(nbpartitionsOpt).toString());
      npos = Integer.parseInt(cmdLine.getValue(nposOpt).toString());
      nneg = Integer.parseInt(cmdLine.getValue(nnegOpt).toString());
      negclass = Integer.parseInt(cmdLine.getValue(negclassOpt).toString());
      
      if(resampling.equalsIgnoreCase("overs")){
        withOversampling = true;  
      }else if(resampling.equalsIgnoreCase("unders")){
    	withUndersampling = true;  
      }else if(resampling.equalsIgnoreCase("smote")){
    	withSmote = true;   
      }else if(resampling.equalsIgnoreCase("hybrid")){
    	withHybrid = true;  
      }
      
      if (cmdLine.hasOption(timeOpt)) {
        preprocessingTimeIsStored = true;  
        timeName = cmdLine.getValue(timeOpt).toString();
      } 
      
      if (log.isDebugEnabled()) {
        log.debug("data : {}", dataName);
        log.debug("dataset : {}", datasetName);         
        log.debug("time : {}", timeName);
        log.debug("Oversampling : {}", withOversampling);
        log.debug("Undersampling : {}", withUndersampling);
        log.debug("SMOTE : {}", withSmote);
        log.debug("Hybrid : {}", withHybrid);
      }
      
      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      dataPreprocessingPath = new Path(dataPreprocessing);
      if(preprocessingTimeIsStored)
        timePath = new Path(timeName);
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    if(withOversampling){
      overSampling();  
    }else if(withUndersampling){	
      underSampling();
    }else if(withSmote){
      smote();
    }else if(withHybrid){
      hybrid();
    }
	
	return 0;
  }
  
  private void hybrid() throws IOException, ClassNotFoundException, InterruptedException {
	preprocessingTime = System.currentTimeMillis();
	
	log.info("Hybrid");
	
	Configuration conf = getConf();	
	
	HybridBuilder hibridBuilder = new HybridBuilder(dataPreprocessingPath, dataPath, datasetPath, conf);
	
	log.info("Building the new data...");
	
	hibridBuilder.build();
	
    preprocessingTime = System.currentTimeMillis() - preprocessingTime;
	
	if(preprocessingTimeIsStored)       
      writeToFilePreprocessingTime(DFUtils.elapsedTime(preprocessingTime));
      
    log.info("Resampling Time: {}", DFUtils.elapsedTime(preprocessingTime)); 
  }
  
  private void smote() throws IOException, ClassNotFoundException, InterruptedException {
    preprocessingTime = System.currentTimeMillis();
	
	log.info("SMOTE");
	
	Configuration conf = getConf();	
	
	SmoteBuilder smoteBuilder = new SmoteBuilder(dataPreprocessingPath, dataPath, datasetPath, conf, partitions);
	
	log.info("Building the new data...");
	
	smoteBuilder.build();
	
    preprocessingTime = System.currentTimeMillis() - preprocessingTime;
	
	if(preprocessingTimeIsStored)       
      writeToFilePreprocessingTime(DFUtils.elapsedTime(preprocessingTime));
      
    log.info("Resampling Time: {}", DFUtils.elapsedTime(preprocessingTime));
    
  }
  
  private void underSampling() throws IOException, ClassNotFoundException, InterruptedException {
	preprocessingTime = System.currentTimeMillis();
	
	log.info("UnderSampling");
	
	Configuration conf = getConf();	
	
	UndersamplingBuilder_v2 undersamplingBuilder = new UndersamplingBuilder_v2(dataPreprocessingPath, dataPath, datasetPath, conf);
	
	log.info("Building the new data...");
	
	undersamplingBuilder.build();
	
	preprocessingTime = System.currentTimeMillis() - preprocessingTime;
	
	if(preprocessingTimeIsStored)       
      writeToFilePreprocessingTime(DFUtils.elapsedTime(preprocessingTime));
      
    log.info("Resampling Time: {}", DFUtils.elapsedTime(preprocessingTime));
  }
  
  private void overSampling() throws IOException, ClassNotFoundException, InterruptedException {
		
	log.info("OverSampling");
	  
	preprocessingTime = System.currentTimeMillis();
	
    Configuration conf = getConf();
	
    OversamplingBuilder oversamplingBuilder = new OversamplingBuilder(dataPreprocessingPath, dataPath, datasetPath, conf, npos, nneg, negclass);
  
    log.info("Building the new data...");
    
    oversamplingBuilder.build();
    
    preprocessingTime = System.currentTimeMillis() - preprocessingTime;
    
    if(preprocessingTimeIsStored)       
      writeToFilePreprocessingTime(DFUtils.elapsedTime(preprocessingTime));
          
    log.info("Resampling Time: {}", DFUtils.elapsedTime(preprocessingTime));
  }
  
  private void writeToFilePreprocessingTime(String time) throws IOException{	
    FileSystem outFS = timePath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;		
	Path filenamePath = new Path(timePath, dataName + "_resampling_time").suffix(".txt");
	try    
	  {	        	
        if (ofile == null) {
	      // this is the first value, it contains the name of the input file
	      ofile = outFS.create(filenamePath);
		  // write the Preprocessing Time	      	      	      	      
		  StringBuilder returnString = new StringBuilder(200);	      
	      returnString.append("=======================================================").append('\n');
		  returnString.append("Resampling Time\n");
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
    ToolRunner.run(new Configuration(), new Resampling(), args);
  }

}
