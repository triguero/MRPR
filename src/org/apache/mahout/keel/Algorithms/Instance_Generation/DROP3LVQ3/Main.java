//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 3-10-2005.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.DROP3LVQ3;

public class Main {

  public static void main (String args[]) {

    DROP3LVQ3 ssma;

    if (args.length != 1)
      System.err.println("Error. A parameter is only needed.");
    else {
      ssma = new DROP3LVQ3 (args[0]);
      ssma.Script = args[0];
      ssma.ejecutar();
    }
  }
}
