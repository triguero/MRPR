package org.apache.mahout.classifier.df.resampling.tools;

public class Referencia implements Comparable {

  //values of the reference
  public int entero;
  public double real;

  /**
   * Default builder
   */
  public Referencia () {} 

  /**
   * Builder
   *
   * @param a Integer value
   * @param b Double value
   */
  public Referencia (int a, double b) {
    entero = a;
	real = b;
  }

  /**
   * Compare to Method
   *
   * @param o1 Reference to compare
   *
   * @return Relative order between the references
   */
  public int compareTo (Object o1) {
	if (this.real > ((Referencia)o1).real)
	  return -1;
	else if (this.real < ((Referencia)o1).real)
	  return 1;
	else return 0;
  }

  /**
   * To String Method
   *
   * @return String representation of the chromosome
   */
  public String toString () {
	return new String ("{"+entero+", "+real+"}");
  }
}
