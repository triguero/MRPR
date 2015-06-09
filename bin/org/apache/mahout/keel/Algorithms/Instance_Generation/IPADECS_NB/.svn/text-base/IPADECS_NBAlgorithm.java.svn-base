/**
	IPADECS_NB.java
	Isaac Triguero Velazquez.
	
	Created by Isaac Triguero Velazquez  23-7-2009
	Copyright (c) 2008 __MyCompanyName__. All rights reserved.
**/

package org.apache.mahout.keel.Algorithms.Instance_Generation.IPADECS_NB;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * IPADECS_NB algorithm calling.
 * @author Isaac Triguero
 */
public class IPADECS_NBAlgorithm extends PrototypeGenerationAlgorithm<IPADECS_NBGenerator>
{
    /**
     * Builds a new IPADECS algorithm
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected IPADECS_NBGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new IPADECS_NBGenerator(train, params);    
    }
    
     /**
     * Main method. Executes IPADECS_NB algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        IPADECS_NBAlgorithm isaak = new IPADECS_NBAlgorithm();
        isaak.executeNB(args);
    }
}
