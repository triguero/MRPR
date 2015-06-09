package org.apache.mahout.classifier.smo.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;


/**
 * Print an ArrayList of predictions.
 */
public class MapredOutput implements Writable, Cloneable {

  //private int [] Predictions; 
  private ArrayList<Integer> Predictions;
  private int numClases;
 // private int[] predictions;

  public MapredOutput() {
  }

  // constructor básico
  public MapredOutput(ArrayList<Integer> Predictions, int numClases) { //, int[] predictions
    this.Predictions = Predictions;
    this.numClases= numClases;

  }
 

  public ArrayList<Integer> getPredictions() {
    return Predictions;
  }

  public int getNumClases(){
	  return numClases;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readPredictions = in.readBoolean();
     
    if (readPredictions) {
        numClases=in.readInt();
        int size=in.readInt();
        //System.out.println("Leyendo: "+size);
       	Predictions = new ArrayList<Integer>();
    	
       	for(int i=0; i< size; i++){
       		Predictions.add(in.readInt());
       	}
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(Predictions != null);
    if (Predictions != null) {
    	out.writeInt(numClases);
    	//System.out.println("Escribiendo: "+Predictions.length);
    	out.writeInt(Predictions.size());
    	for(int i=0; i<Predictions.size();i++){
    		out.writeInt(Predictions.get(i));
    	}
      
    }
  }

  @Override
  public MapredOutput clone() {
    return new MapredOutput(Predictions,numClases); 
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;

    return ((Predictions == null && mo.getPredictions() == null) || (Predictions != null && Predictions.equals(mo.getPredictions()))); 
  }


}

