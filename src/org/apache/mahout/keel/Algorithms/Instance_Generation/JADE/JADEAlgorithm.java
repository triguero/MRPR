//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.JADE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * JADE algorithm calling.
 * @author Isaac Triguero
 */
public class JADEAlgorithm extends PrototypeGenerationAlgorithm<JADEGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected JADEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new JADEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes JADE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        JADEAlgorithm isaak = new JADEAlgorithm();
        isaak.execute(args);
    }
}
