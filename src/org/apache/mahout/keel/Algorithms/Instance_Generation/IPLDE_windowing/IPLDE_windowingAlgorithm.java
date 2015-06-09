/**
	IPLDE_windowing.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDE_windowing;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPLDE_windowing algorithm calling.
 * @author Isaac Triguero
 */
public class IPLDE_windowingAlgorithm extends PrototypeGenerationAlgorithm<IPLDE_windowingGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPLDE_windowingGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPLDE_windowingGenerator(train, params);    
    }
    
     /**
     * Main method. Executes IPLDE_windowing algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPLDE_windowingAlgorithm isaak = new IPLDE_windowingAlgorithm();
        isaak.execute(args);
    }
}
