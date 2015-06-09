package org.apache.mahout.classifier.pg.utils;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.pg.data.DataConverter;
import org.apache.mahout.classifier.pg.data.Instance;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Utility class that contains various helper methods
 */
public final class PGUtils {
  public PGUtils() { }
  
  /**
   * Writes an RuleBase[] into a DataOutput
   * @throws java.io.IOException
   
  public static void writeArray(DataOutput out, RuleBase[] array) throws IOException {
    out.writeInt(array.length);
    for (RuleBase w : array) {
      w.write(out);
    }
  }
  */
  /**
   * Reads a RuleBase[] from a DataInput
   * @throws java.io.IOException
   
  public static RuleBase[] readRuleBasesArray(DataInput in) throws IOException {
    int length = in.readInt();
    RuleBase[] ruleBases = new RuleBase[length];
    for (int index = 0; index < length; index++) {
      RuleBase ruleBase = new RuleBase();
      ruleBase.readFields(in);
      ruleBases[index] = ruleBase;
    }
    
    return ruleBases;
  }
  */
  
  
  public static InstanceSet readHeader(FileSystem fs, Path fpath) throws IOException{
	    FSDataInputStream input = fs.open(fpath);
	    Scanner scanner = new Scanner(input, "UTF-8");
	    
	    ArrayList<String> cabecera = new ArrayList();
	    while (scanner.hasNextLine()) {
	        String line = scanner.nextLine();
	        cabecera.add(line);
	    }
	    
	   scanner.close();
	    
	   Attributes.clearAll();//BUGBUGBUG
	   InstanceSet training = new InstanceSet();     
	      
	   training.parseHeaderFromString(cabecera,true);
	//   training.setAttributesAsNonStatic();
     //  InstanceAttributes att = InstanceSet.getAttributeDefinitions();
      // Prototype.setAttributesTypes(att);  
        
       return training;
  }
  
  
  
  public static String readHeader(String cabecera) throws IOException{
	   Attributes.clearAll();//BUGBUGBUG
	   InstanceSet training = new InstanceSet();     
	      
	   ArrayList<String> header = new ArrayList<String>();
	   
	   String parts[]= cabecera.split("@");
	   
	   for(int i=0; i<parts.length;i++){
		   header.add("@"+parts[i]);
		   //System.out.println(parts[i]);
	   }
			   
	   training.parseHeaderFromString(header,true);
	   training.setAttributesAsNonStatic();
       InstanceAttributes att = InstanceSet.getAttributeDefinitions();
       Prototype.setAttributesTypes(att);  
        
       return cabecera;
  }
  
  /**
   * This method read the header file, to initialize the Attribute and Instance Classes of KEEL.
   * FROM LOCAL DISK
   * @param nameOfFile
   * @return
   */
  public static PrototypeSet readPrototypeSet(String nameOfFile)
  {
      Attributes.clearAll();//BUGBUGBUG
      InstanceSet training = new InstanceSet();        
      try
      {
      	//System.out.print("PROBANDO:\n"+nameOfFile);
          training.readSet(nameOfFile, true) ;
          training.setAttributesAsNonStatic();
          InstanceAttributes att = training.getAttributeDefinitions();
          Prototype.setAttributesTypes(att);            
      }
      catch(Exception e)
      {
          System.err.println("readPrototypeSet has failed!");
          e.printStackTrace();
      }
	return new PrototypeSet(training);
  }
  

  
  
  /**
   * Writes a double[] into a DataOutput
   * @throws java.io.IOException
   */
  public static void writeArray(DataOutput out, double[] array) throws IOException {
    out.writeInt(array.length);
    for (double value : array) {
      out.writeDouble(value);
    }
  }
  
  /**
   * Reads a double[] from a DataInput
   * @throws java.io.IOException
   */
  public static double[] readDoubleArray(DataInput in) throws IOException {
    int length = in.readInt();
    double[] array = new double[length];
    for (int index = 0; index < length; index++) {
      array[index] = in.readDouble();
    }
    
    return array;
  }
  
  /**
   * Writes an int[] into a DataOutput
   * @throws java.io.IOException
   */
  public static void writeArray(DataOutput out, int[] array) throws IOException {
    out.writeInt(array.length);
    for (int value : array) {
      out.writeInt(value);
    }
  }
  
  /**
   * Reads an int[] from a DataInput
   * @throws java.io.IOException
   */
  public static int[] readIntArray(DataInput in) throws IOException {
    int length = in.readInt();
    int[] array = new int[length];
    for (int index = 0; index < length; index++) {
      array[index] = in.readInt();
    }
    
    return array;
  }
  
  /**
   * Return a list of all files in the output directory
   * @throws IOException if no file is found
   */
  public static Path[] listOutputFiles(FileSystem fs, Path outputPath) throws IOException {
    List<Path> outputFiles = Lists.newArrayList();
    for (FileStatus s : fs.listStatus(outputPath, PathFilters.logsCRCFilter())) {
      if (!s.isDir() && !s.getPath().getName().startsWith("_")) {
        outputFiles.add(s.getPath());
      }
    }
    if (outputFiles.isEmpty()) {
      throw new IOException("No output found !");
    }
    return outputFiles.toArray(new Path[outputFiles.size()]);
  }

  /**
   * Formats a time interval in milliseconds to a String in the form "hours:minutes:seconds:millis"
   */
  public static String elapsedTime(long milli) {
    long seconds = milli / 1000;
    milli %= 1000;
    
    long minutes = seconds / 60;
    seconds %= 60;
    
    long hours = minutes / 60;
    minutes %= 60;
    
    return hours + "h " + minutes + "m " + seconds + "s " + milli;
  }

  
  public static String elapsedSeconds(long milli) {
	    double seconds = milli / 1000.;
   
	    return seconds + " seconds ";
  }
  
  public static void storeWritable(Configuration conf, Path path, Writable writable) throws IOException {
    FileSystem fs = path.getFileSystem(conf);

    FSDataOutputStream out = fs.create(path);
    try {
      writable.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }
  }
}
