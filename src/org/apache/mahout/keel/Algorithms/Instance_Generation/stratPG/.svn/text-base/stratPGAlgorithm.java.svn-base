/**
	stratPG.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.stratPG;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * stratPG algorithm calling.
 * @author Isaac Triguero
 */
public class stratPGAlgorithm extends PrototypeGenerationAlgorithm<stratPGGenerator>
{
    /**
     * Builds a new IPADE algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected stratPGGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new stratPGGenerator(train, params);    
    }
    
     /**
     * Main method. Executes stratPG algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        stratPGAlgorithm isaak = new stratPGAlgorithm();
        isaak.execute(args);
    }
}
