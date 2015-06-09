package org.apache.mahout.keel.Algorithms.Instance_Generation.Basic;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import org.apache.mahout.classifier.pg.data.*;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.Instance;
import org.apache.mahout.keel.Dataset.InstanceAttributes;
import org.apache.mahout.keel.Dataset.InstanceSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.*;

import org.apache.mahout.keel.Algorithms.Instance_Generation.SSMASFLSDE.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.SSMASFLSDE_windowing.SSMASFLSDE_windowing;


public class HandlerSSMASFLSDE {
		
	private int[][] predictions;
	private String algSufix = "";

	public PrototypeSet reducedSet;
	

	public void ejecutar(Data data, Context context)throws Exception{
		PrototypeSet trainPG=new PrototypeSet(data);
	    context.progress();

	    trainPG.print();
		ejecutar(trainPG.toInstanceSet(),trainPG, context);
	}

	public PrototypeSet ejecutar(InstanceSet train, PrototypeSet trainPG, Context context) throws Exception{
		// ejecutar el metodo
	  //  Attributes.clearAll();
	    String[] argumentos = new String[1];
	    argumentos[0] = "NOFILE";
	    System.out.println("Size of training para HandleSSMA"+trainPG.size());
	    System.err.println("Size of training para HandleSSMA"+trainPG.size());

		SSMASFLSDE ssma = new SSMASFLSDE (argumentos[0],train, context);
	    ssma.Script = argumentos[0];
	    
	    ssma.establishTrain(trainPG);
	    ssma.establishContext(context);
	    
	    context.progress();
	    ssma.ejecutar();
	
	    context.progress();
	    reducedSet = ssma.resultingSet;
	
		return reducedSet;
		
	}
	
	public PrototypeSet ejecutarWindowing(InstanceSet train, PrototypeSet trainPG, Context context, int strataSize) throws Exception{
		// ejecutar el metodo
	  //  Attributes.clearAll();
	    String[] argumentos = new String[1];
	    argumentos[0] = "NOFILE";
		
		SSMASFLSDE_windowing ssma = new SSMASFLSDE_windowing (argumentos[0],train, context, strataSize);
	    ssma.Script = argumentos[0];
	    ssma.establishTrain(trainPG);
	    ssma.establishContext(context);
	    
	    context.progress();
	    ssma.ejecutar();
	
	    context.progress();
	    reducedSet = ssma.resultingSet;
	
		return reducedSet;
		
	}
	
		

	
}
