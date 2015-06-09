package org.apache.mahout.classifier.pg.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;


// Print a PrototypeSet! .

/**
 * Print a reduced set as a PrototypeSet.
 */
public class MapredOutput implements Writable, Cloneable {

  private PrototypeSet RS;  // conjunto reducido.

 // private int[] predictions;

  public MapredOutput() {
  }

  // constructor básico
  public MapredOutput(PrototypeSet RS) { //, int[] predictions
    this.RS = RS;
  //  this.predictions = predictions;
  }
 
  // constructor sin predicciones.
 /* public MapredOutput(PrototypeSet RS) {
    this(RS, null);
  }
*/
  public PrototypeSet getRS() {
    return RS;
  }

  /*int[] getPredictions() {
    return predictions;
  }
*/
  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readRuleBase = in.readBoolean();
    if (readRuleBase) {
      PrototypeSet rs = new PrototypeSet();
      rs.readFields(in);
      RS = rs;	
    }
/*
    boolean readPredictions = in.readBoolean();
    if (readPredictions) {
      //predictions = Chi_RWUtils.readIntArray(in);
    }
    */
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(RS != null);
    if (RS != null) {
      RS.write(out);
    }

    /*
    out.writeBoolean(predictions != null);
    if (predictions != null) {
     // Chi_RWUtils.writeArray(out, predictions);
    }
    */
  }

  @Override
  public MapredOutput clone() {
    return new MapredOutput(RS); //, predictions
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

    return ((RS == null && mo.getRS() == null) || (RS != null && RS.equals(mo.getRS()))); //&& Arrays.equals(predictions, mo.getPredictions()
  }

  /*
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

