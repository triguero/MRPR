package org.apache.mahout.keel.Algorithms.Preprocess.Missing_Values.MVI_DE;

import java.util.*;
import java.io.*;
import org.apache.mahout.keel.Dataset.InstanceSet;


/**
 * <p>
 * Main class to parse the parameters of the algorithm
 * </p>
 */
public class Parameters {
	
	private static BufferedReader br;
	
	// parameters of the algorithm ------------------
	public static String algorithmName;
	
	public static String trainInputFile;
	public static String testInputFile;
	
	public static String trainOutputFile;
	public static String testOutputFile;
	public static String logOutputFile;
	
	//-----------------------------------------------
	public static double ScalingFactor;
    static public int maxIterations;
    static public OptimizedNN ONN;
    static public boolean iterativeHVDM;
    
	// parameters of the algorithm ------------------
	public static long seed;
	
	public static InstanceSet IS_tra;
	public static InstanceSet IS_tst;
	
	public static int numClasses;
	
	public static int numAttributes;
	public static int[] attributeType;
	public static boolean[][] missingTRA;
	public static boolean[][] missingTST;
	
	public static int numInstancesTRA;
	public static int numInstancesTST;
	
	public static double[][] instanceTRA;
	public static double[][] instanceTST;
	
	public static int[] classTRA;
	public static int[] classTST;
	
    static public int[] numNominalValuesAtt;
    public static int[][] nominalIntTRA;
	public static int[][] nominalIntTST;
	
	public static double[][] integerValueTRA;
	public static double[][] integerValueTST;
	
	// parametros C45
    static public boolean prune;
    static public double confidence;
    static public int itemsetsPerLeaf;
    
    static public String base;
    public static InstanceSet IS_tra_C45;
	// ----------------------------------------------
	

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It parses the parameters of the algorithm
	 * </p>
	 * @param fileName the configuration file name
	 */
	public static void doParse(String fileName){
		
		try {
			br = new BufferedReader(new FileReader(fileName));
		}catch(Exception e){
			e.printStackTrace();
			System.exit(1);
		}

		parseParameters();
		System.out.println ("Parameters correctly parsed");
	}
	
//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It parses the header of the ARFF file
	 * </p>
	 */
	static void parseParameters(){
		
		String str = getLine();
		
		while(str != null){
			
			System.out.println("Line: " + str);
			StringTokenizer st = new StringTokenizer(str,"=");
			String name = st.nextToken();
			name = name.trim();

			if(name.equalsIgnoreCase("algorithm")) 
				processAlgorithmName(st);
			
			else if(name.equalsIgnoreCase("inputData"))
				processInputs(st);
			
			else if(name.equalsIgnoreCase("outputData"))
				processOutputs(st);
			
			else processParameters(st,name);

			str = getLine();
		}
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It processes the algorithm name
	 * </p>
	 * @param st string with the algorithm name
	 */
	static void processAlgorithmName(StringTokenizer st){
		
		if(!st.hasMoreTokens()){
			System.err.println("Parse error processing algorithm name");
			System.exit(1);
		}
		
		String name = st.nextToken();
		name = name.trim();
		System.out.println("The name is: " + name);
		
		if(!validateAlgorithmName(name)){
			System.err.println("This config file is not for us: " + name);
			System.exit(1);
		}
		
		algorithmName = new String(name);
		System.out.println ("Returning from processAlgorithmName");
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It processes input files
	 * </p>
	 * @param st string with the input files
	 */
	static void processInputs(StringTokenizer st){
		
		if(!st.hasMoreTokens()){
			System.err.println("Parse error processing inputs");
			System.exit(1);
		}
		
		String files = st.nextToken();
		files = files.trim();
		
		if(!(files.startsWith("\"") && files.endsWith("\""))){
			System.err.println("Parse error processing inputs " + files);
			System.exit(1);
		}
		
		files.replaceAll("^\"",""); files.replaceAll("\"$","");
		StringTokenizer st2 = new StringTokenizer(files,"\"");
		
		try {
			String file1 = st2.nextToken();
			st2.nextToken();
			String file2 = st2.nextToken();
			insertStringParameter("trainInputFile",file1);
			insertStringParameter("testInputFile",file2);
		}catch(NoSuchElementException e){
			System.err.println("Parse error processing inputs " + files);
			System.exit(1);
		}
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It processes output files
	 * </p>
	 * @param st string with the output files
	 */
	static void processOutputs(StringTokenizer st){
		
		if(!st.hasMoreTokens()){
			System.err.println("Parse error processing outputs");
			System.exit(1);
		}
		
		String files = st.nextToken();
		files = files.trim();
		
		if(!(files.startsWith("\"") && files.endsWith("\""))){
			System.err.println("Parse error processing outputs " + files);
			System.exit(1);
		}
		
		files.replaceAll("^\"",""); files.replaceAll("\"$","");
		StringTokenizer st2 = new StringTokenizer(files,"\"");
		
		try {
			String file1 = st2.nextToken();
			st2.nextToken();
			String file2 = st2.nextToken();
			st2.nextToken();
			String file3 = st2.nextToken();
			insertStringParameter("trainOutputFile",file1);
			insertStringParameter("testOutputFile",file2);
			insertStringParameter("logOutputFile",file3);
		}catch(NoSuchElementException e){
			System.err.println("Parse error processing outputs " + files);
			System.exit(1);
		}
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It processes the parameters
	 * </p>
	 * @param st string with the parameter value
	 * @param paramName parameter name
	 */
	static void processParameters(StringTokenizer st, String paramName){
		
		if(!st.hasMoreTokens()){
			System.err.println("Parse error processing parameter " + paramName);
			System.exit(1);
		}
		
		String paramValue = st.nextToken();
		paramValue = paramValue.trim();

		System.out.println("paramName: " + paramName);
		
		if(isReal(paramName)) 
			insertRealParameter(paramName,paramValue);
		
		else if(isInteger(paramName))
			insertIntegerParameter(paramName,paramValue);
		
		else if(isBoolean(paramName))
			insertBooleanParameter(paramName,paramValue);
		
		else if(isString(paramName))
			insertStringParameter(paramName,paramValue);
		
		else{
			System.err.println("Unknown parameter " + paramName);
			System.exit(1);
		}
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It parses a real parameter
	 * </p>
	 * @param paramName parameter name
	 * @return true if the parameter is real, false otherwise
	 */
	static boolean isReal(String paramName){
		
		if(paramName.equalsIgnoreCase("ScalingFactor")) return true;
		if(paramName.equalsIgnoreCase("confidence")) return true;
		
		return false;
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It parses an integer parameter
	 * </p>
	 * @param paramName parameter name
	 * @return true if the parameter is integer, false otherwise
	 */
	static boolean isInteger(String paramName){

		if(paramName.equalsIgnoreCase("maxIterations")) return true;
		if(paramName.equalsIgnoreCase("seed")) return true;
		if(paramName.equalsIgnoreCase("itemsetsPerLeaf")) return true;
		
		return false;
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It parses an boolean parameter
	 * </p>
	 * @param paramName parameter name
	 * @return true if the parameter is boolean, false otherwise
	 */
	static boolean isBoolean(String paramName){
		
		if(paramName.equalsIgnoreCase("iterativeHVDM")) return true;
		if(paramName.equalsIgnoreCase("prune")) return true;
		
		return false;
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It parses a string parameter
	 * </p>
	 * @param paramName parameter name
	 * @return true if the parameter is string, false otherwise
	 */
	static boolean isString(String paramName){
		
		if(paramName.equalsIgnoreCase("base")) return true;
		
		return false;
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It does the assignment of a real parameter
	 * </p>
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	static void insertRealParameter(String paramName, String paramValue){
		
		double num = Double.parseDouble(paramValue);
		
		try {
			Parameters param = new Parameters();
			java.lang.reflect.Field f = Parameters.class.getField(paramName);
			f.setDouble(param,num);
		}catch(Exception e){
			System.err.println("Cannot set param " + paramName);
			System.exit(1);
		}
	}

//*******************************************************************************************************************************
	
	/**
	 * <p>
	 * It does the assignment of an integer parameter
	 * </p>
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	static void insertIntegerParameter(String paramName, String paramValue){
		
		int num = Integer.parseInt(paramValue);
		
		try{
			Parameters param = new Parameters();
			java.lang.reflect.Field f = Parameters.class.getField(paramName);
			f.setInt(param,num);
		}catch(Exception e){
			System.err.println("Cannot set param " + paramName);
			System.exit(1);
		}
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It does the assignment of a boolean parameter
	 * </p>
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	static void insertBooleanParameter(String paramName, String paramValue){
		
		boolean val = false;
		
		if(paramValue.equals("true")) val = true;

		try{
			Parameters param = new Parameters();
			java.lang.reflect.Field f = Parameters.class.getField(paramName);
			f.setBoolean(param,val);
		}catch(Exception e){
			System.err.println("Cannot set param " + paramName);
			System.exit(1);
		}
	}

//*******************************************************************************************************************************

	/**
	 * <p>
	 * It does the assignment of a string parameter
	 * </p>
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	static void insertStringParameter(String paramName, String paramValue){
		
		try{
			Parameters param = new Parameters();
			java.lang.reflect.Field f = Parameters.class.getField(paramName);
			f.set(param,new String(paramValue));
		}catch(Exception e){
			System.err.println("Cannot set param " + paramName);
			System.exit(1);
		}
	}
	
//*******************************************************************************************************************************

	/**
	 * <p>
	 * It checks the algorithm name
	 * </p>
	 * @param name algorithm name
	 * @return true if the algorithm name is correct, false otherwise
	 */
	static boolean validateAlgorithmName(String name){

		if(name.equalsIgnoreCase("MVI_DE")) return true;

		return false;
	}

//*******************************************************************************************************************************

	/**
	 * <p>
	 * It returns the next line in the configuration file
	 * </p>
	 * @return the next line in the configuration file
	 */
	static String getLine(){
		
		String st = null;
		
		do{
			
			try {
				st = br.readLine();
			}catch(Exception e){
				e.printStackTrace();
				System.exit(1);
			}
		
		}while(st != null && st.equals(""));
		
		return st;
	}
	
}