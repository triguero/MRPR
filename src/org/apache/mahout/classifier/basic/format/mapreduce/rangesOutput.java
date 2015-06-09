package org.apache.mahout.classifier.basic.format.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Pair;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;


// Print a PrototypeSet! .

/**
 * Print a reduced set as a PrototypeSet.
 */
public class rangesOutput implements Writable, Cloneable {

  public double mins[];
  public double maxs[]; 

 // private int[] predictions;

  public rangesOutput() {
  }

  // constructor básico
  public rangesOutput(double mins[], double maxs[]) { //, int[] predictions
    this.mins = mins;
    this.maxs = maxs;
  }
 
  public Pair<double[],double[]> getMaxMins(){
	  Pair<double[],double[]> maxmins = new Pair(mins,maxs);
	  return maxmins;
	}

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readRuleBase = in.readBoolean();
    if (readRuleBase) {
      int size= in.readInt();
    	mins=new double[size];
      for(int i=0;i<size;i++){
    	  mins[i]=in.readDouble();
      }
    	maxs=new double[size];
      for(int i=0;i<size;i++){
    	  maxs[i]=in.readDouble();
      }

    }

  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(mins != null);
    if (mins != null) {
      out.writeInt(mins.length);
      for(int i=0;i<mins.length;i++){
    	  out.writeDouble(mins[i]);
      }
      for(int i=0;i<mins.length;i++){
    	  out.writeDouble(maxs[i]);
      }
    	
    }


  }

  @Override
  public rangesOutput clone() {
    return new rangesOutput(mins,maxs); //, predictions
  }

  
  /*
   * @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;

    return ((RS == null && mo.getRS() == null) || (RS != null && RS.equals(mo.getRS()))); //&& Arrays.equals(predictions, mo.getPredictions()
  }

  
  @Override
  public int hashCode() {
    int hashCode = RS == null ? 1 : RS.hashCode();
    for (int prediction : predictions) {
      hashCode = 31 * hashCode + prediction;
    }
    return hashCode;
  }
  

  @Override
  public String toString() {
    return "{" + RS + " | " + Arrays.toString(predictions) + '}';
  }
  */

}

