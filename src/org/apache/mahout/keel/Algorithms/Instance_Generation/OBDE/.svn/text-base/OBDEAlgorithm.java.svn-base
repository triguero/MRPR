//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.OBDE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * OBDE algorithm calling.
 * @author Isaac Triguero
 */
public class OBDEAlgorithm extends PrototypeGenerationAlgorithm<OBDEGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected OBDEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new OBDEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes OBDE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        OBDEAlgorithm isaak = new OBDEAlgorithm();
        isaak.execute(args);
    }
}
