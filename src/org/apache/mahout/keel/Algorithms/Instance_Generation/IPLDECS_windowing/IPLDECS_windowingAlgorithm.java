/**
	IPLDECS_windowing.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS_windowing;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPLDECS_windowing algorithm calling.
 * @author Isaac Triguero
 */
public class IPLDECS_windowingAlgorithm extends PrototypeGenerationAlgorithm<IPLDECS_windowingGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPLDECS_windowingGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPLDECS_windowingGenerator(train, params);    
    }
    
    
    
    
     /**
     * Main method. Executes IPLDECS_windowing algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPLDECS_windowingAlgorithm isaak = new IPLDECS_windowingAlgorithm();
        isaak.execute(args);
    }
}
