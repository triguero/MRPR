//
//  Main.java
//
//  Salvador García López
//
//  Created by Salvador García López 10-7-2004.
//  Copyright (c) 2004 __MyCompanyName__. All rights reserved.
//

package org.apache.mahout.keel.Algorithms.Instance_Generation.DE;

import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeGenerationAlgorithm;
import org.apache.mahout.keel.Algorithms.Instance_Generation.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.*;

import java.util.*;


import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * DE algorithm calling.
 * @author Isaac Triguero
 */
public class DEReducer extends Reducer<Text, IntWritable, Text, IntWritable>
{

    public void reduce()
    {


    }
}
