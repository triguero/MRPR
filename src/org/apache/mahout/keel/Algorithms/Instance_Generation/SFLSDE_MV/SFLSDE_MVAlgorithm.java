//
//  Main.java
//
//  Salvador Garc�a L�pez
//
//  Created by Salvador Garc�a L�pez 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.SFLSDE_MV;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;

/**
 * SFLSDE_MV algorithm calling.
 * @author Isaac Triguero
 */
public class SFLSDE_MVAlgorithm extends PrototypeGenerationAlgorithm<SFLSDE_MVGenerator>
{
    /**
     * Builds a new ChenGenerator.
     * @param train Training data set.
     * @param params Parameters of the method.
     */
    protected SFLSDE_MVGenerator buildNewPrototypeGenerator(PrototypeSet train, Parameters params)
    {
       return new SFLSDE_MVGenerator(train, params);    
    }
    
     /**
     * Main method. Executes SFLSDE_MV algorithm.
     * @param args Console arguments of the method.
     */
    public static void main(String args[])
    {
        SFLSDE_MVAlgorithm isaak = new SFLSDE_MVAlgorithm();
        isaak.executeNB(args);
    }

	protected SFLSDE_MVGenerator buildNewPrototypeGenerator(PrototypeSet train,
			PrototypeSet test, Parameters params) {
		// TODO Auto-generated method stub
		return null;
	}
}
