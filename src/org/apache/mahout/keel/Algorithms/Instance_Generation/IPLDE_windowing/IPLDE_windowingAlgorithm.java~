/**
	IPLDE.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPLDE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPLDE algorithm calling.
 * @author Isaac Triguero
 */
public class IPLDEAlgorithm extends PrototypeGenerationAlgorithm<IPLDEGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPLDEGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPLDEGenerator(train, params);    
    }
    
     /**
     * Main method. Executes IPLDE algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPLDEAlgorithm isaak = new IPLDEAlgorithm();
        isaak.execute(args);
    }
}
