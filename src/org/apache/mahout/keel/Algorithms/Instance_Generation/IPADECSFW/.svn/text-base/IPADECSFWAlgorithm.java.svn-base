/**
	IPADECSFW.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPADECSFW;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPADECSFW algorithm calling.
 * @author Isaac Triguero
 */
public class IPADECSFWAlgorithm extends PrototypeGenerationAlgorithm<IPADECSFWGenerator>
{
    /**
     * Builds a new IPADECS algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPADECSFWGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPADECSFWGenerator(train, params);    
    }
    
     /**
     * Main method. Executes IPADECSFW algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPADECSFWAlgorithm isaak = new IPADECSFWAlgorithm();
        isaak.executeFeatures(args);
    }
}
