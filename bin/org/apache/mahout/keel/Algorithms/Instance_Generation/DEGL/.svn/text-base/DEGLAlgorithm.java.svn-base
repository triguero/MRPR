//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.DEGL;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * DEGL algorithm calling.
 * @author Isaac Triguero
 */
public class DEGLAlgorithm extends PrototypeGenerationAlgorithm<DEGLGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected DEGLGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new DEGLGenerator(train, params);    
    }
    
     /**
     * Main method. Executes DEGL algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        DEGLAlgorithm isaak = new DEGLAlgorithm();
        isaak.execute(args);
    }
}
