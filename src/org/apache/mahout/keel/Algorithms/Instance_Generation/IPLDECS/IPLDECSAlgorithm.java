/**
	IPLDECS.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDECS;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPLDECS algorithm calling.
 * @author Isaac Triguero
 */
public class IPLDECSAlgorithm extends PrototypeGenerationAlgorithm<IPLDECSGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPLDECSGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPLDECSGenerator(train, params);    
    }
    
    
    
    
     /**
     * Main method. Executes IPLDECS algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPLDECSAlgorithm isaak = new IPLDECSAlgorithm();
        isaak.execute(args);
    }
}
