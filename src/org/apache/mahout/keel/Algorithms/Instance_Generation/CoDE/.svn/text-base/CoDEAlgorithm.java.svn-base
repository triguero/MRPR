//
//  Main.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.CoDE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * CoDE algorithm calling.
 * @author Isaac Triguero
 */
public class CoDEAlgorithm extends PrototypeGenerationAlgorithm<CoDEGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected CoDEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new CoDEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes CoDE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        CoDEAlgorithm isaak = new CoDEAlgorithm();
        isaak.execute(args);
    }
}
