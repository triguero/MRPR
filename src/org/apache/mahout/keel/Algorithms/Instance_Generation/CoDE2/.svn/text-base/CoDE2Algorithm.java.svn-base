//
//  Main.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.CoDE2;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * CoDE2 algorithm calling.
 * @author Isaac Triguero
 */
public class CoDE2Algorithm extends PrototypeGenerationAlgorithm<CoDE2Generator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected CoDE2Generator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new CoDE2Generator(train, params);    
    }
    
     /**
     * Main method. Executes CoDE2 algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        CoDE2Algorithm isaak = new CoDE2Algorithm();
        isaak.execute(args);
    }
}
