package org.apache.mahout.classifier.df.resampling.tools;

public class CheckException extends Exception{
	
 /**
 * Creates a new instance of CheckException
 */
  public CheckException() {
    super();
  }


/**
 * Does instance a new CheckException with the message
 * specified and the Vector with all the errors.
 * @param msg is the message of the exception
 *
 */
  public CheckException(String msg){
    super(msg);
  }
}
