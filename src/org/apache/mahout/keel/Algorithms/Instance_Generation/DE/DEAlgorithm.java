//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.DE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * DE algorithm calling.
 * @author Isaac Triguero
 */
public class DEAlgorithm extends PrototypeGenerationAlgorithm<DEGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected DEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new DEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes DE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        DEAlgorithm isaak = new DEAlgorithm();
        isaak.execute(args);
    }
}
