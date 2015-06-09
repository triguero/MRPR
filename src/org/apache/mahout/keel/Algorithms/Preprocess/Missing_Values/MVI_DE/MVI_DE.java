package org.apache.mahout.keel.Algorithms.Preprocess.Missing_Values.MVI_DE;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Vector;
import org.core.Randomize;
import org.apache.mahout.keel.Algorithms.Genetic_Rule_Learning.Globals.FileManagement;
import org.apache.mahout.keel.Dataset.Attribute;
import org.apache.mahout.keel.Dataset.Attributes;
import org.apache.mahout.keel.Dataset.Instance;
import org.apache.mahout.keel.Dataset.InstanceSet;


public class MVI_DE {
	
	Vector[] indexCla;
	double  current_acc;

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public MVI_DE() throws Exception{
		
		Randomize.setSeed(Parameters.seed);
		
		Parameters.IS_tra = new InstanceSet();
		Parameters.IS_tra.readSet(Parameters.trainInputFile, true);
		
		Parameters.IS_tra_C45 = new InstanceSet();
		Parameters.IS_tra_C45.readSet(Parameters.trainInputFile, false);
		Instance[] ins_tra = Parameters.IS_tra.getInstances();
		
		Parameters.numInstancesTRA = ins_tra.length;
		Parameters.numAttributes = Attributes.getInputNumAttributes();
		Parameters.numClasses = Attributes.getOutputAttribute(0).getNumNominalValues();
		Parameters.instanceTRA = new double[Parameters.numInstancesTRA][Parameters.numAttributes];
		Parameters.attributeType = new int[Parameters.numAttributes];
		Parameters.nominalIntTRA = new int[Parameters.numInstancesTRA][Parameters.numAttributes];
		Parameters.integerValueTRA = new double[Parameters.numInstancesTRA][Parameters.numAttributes];
		Parameters.classTRA = new int[Parameters.numInstancesTRA];
		Parameters.missingTRA = new boolean[Parameters.numInstancesTRA][Parameters.numAttributes];
		Parameters.numNominalValuesAtt = new int[Parameters.numAttributes];
		
		// to normalize the training set
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			
			Parameters.classTRA[i] = ins_tra[i].getOutputNominalValuesInt(0);
			Parameters.missingTRA[i] = ins_tra[i].getInputMissingValues();
			
			for(int a = 0 ; a < Parameters.numAttributes ; ++a){
				
				if(!Parameters.missingTRA[i][a]){
					
					if(Attributes.getAttribute(a).getType() == Attribute.NOMINAL){
						Parameters.attributeType[a] = Attribute.NOMINAL;
						Parameters.numNominalValuesAtt[a] = Attributes.getInputAttribute(a).getNumNominalValues();
						int value = ins_tra[i].getInputNominalValuesInt(a);
						Parameters.nominalIntTRA[i][a] = value;			
						value /= (Attributes.getInputAttribute(a).getNumNominalValues()-1);
						Parameters.instanceTRA[i][a] = value;
					}
					
					if(Attributes.getAttribute(a).getType() == Attribute.INTEGER){
						Parameters.attributeType[a] = Attribute.INTEGER;
						double max = Attributes.getInputAttribute(a).getMaxAttribute();
						double min = Attributes.getInputAttribute(a).getMinAttribute();
						double value = ins_tra[i].getInputRealValues(a);
						value -= min;
						value /= (max-min);
						Parameters.instanceTRA[i][a] = value;
						Parameters.integerValueTRA[i][a] = value;
					}
					
					if(Attributes.getAttribute(a).getType() == Attribute.REAL){
						Parameters.attributeType[a] = Attribute.REAL;
						double max = Attributes.getInputAttribute(a).getMaxAttribute();
						double min = Attributes.getInputAttribute(a).getMinAttribute();
						double value = ins_tra[i].getInputRealValues(a);
						value -= min;
						value /= (max-min);
						Parameters.instanceTRA[i][a] = value;
					}
				}
				
			}
			
		}

		//compute instances per class
		indexCla = new Vector[Parameters.numClasses];
		
		for(int c = 0 ; c < Parameters.numClasses ; ++c)
			indexCla[c] = new Vector();
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			indexCla[Parameters.classTRA[i]].add(i);
		}

	}			
		
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
							
	public double Denormalize(double value, int att){
		
		if(Parameters.attributeType[att] == Attribute.REAL){
			double min = Attributes.getInputAttribute(att).getMinAttribute();
			double max = Attributes.getInputAttribute(att).getMaxAttribute();
			return (value*(max-min))+min;
		}
		
		if(Parameters.attributeType[att] == Attribute.INTEGER){
			double min = Attributes.getInputAttribute(att).getMinAttribute();
			double max = Attributes.getInputAttribute(att).getMaxAttribute();
			double v = (value*(max-min))+min;
			return Math.rint(v);
			
		}
		
		if(Parameters.attributeType[att] == Attribute.NOMINAL){
			double res = value*(Attributes.getInputAttribute(att).getNumNominalValues()-1);
			return Math.rint(res);
		}
		
		return -1;
	}

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public double Normalize(double value, int att){
		
		if(Parameters.attributeType[att] == Attribute.REAL){
			double min = Attributes.getInputAttribute(att).getMinAttribute();
			double max = Attributes.getInputAttribute(att).getMaxAttribute();
			double norm = (value-min)/(max-min);
			return norm;
		}
		
		if(Parameters.attributeType[att] == Attribute.INTEGER){
			double min = Attributes.getInputAttribute(att).getMinAttribute();
			double max = Attributes.getInputAttribute(att).getMaxAttribute();
			double norm = (value-min)/(max-min);
			return norm;
			
		}
		
		if(Parameters.attributeType[att] == Attribute.NOMINAL){
			double res = value/(Attributes.getInputAttribute(att).getNumNominalValues()-1);
			return res;
		}
		
		return -1;
	}
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public void CreateTrainingFile(){

		// to create the train file-----------------------------------------
		String header = "";
		header = "@relation " + Attributes.getRelationName() + "\n";
		header += Attributes.getInputAttributesHeader();
		header += Attributes.getOutputAttributesHeader();
		header += Attributes.getInputHeader() + "\n";
		header += Attributes.getOutputHeader() + "\n";
		header += "@data\n";
		
		FileManagement fm = new FileManagement();
		
		try {
			
			fm.initWrite(Parameters.trainOutputFile);
			fm.writeLine(header);
			
			
			for(int k = 0 ; k < Parameters.numInstancesTRA ; k++){
								
				String newInstance = "";
				
				for(int j = 0 ; j < Parameters.numAttributes ; j++){
					
					if(Parameters.missingTRA[k][j]){
						
						if(Parameters.attributeType[j] == Attribute.REAL)
							newInstance += Denormalize(Parameters.instanceTRA[k][j],j);
						if(Parameters.attributeType[j] == Attribute.INTEGER)
							newInstance += (int) Denormalize(Parameters.instanceTRA[k][j],j);
						if(Parameters.attributeType[j] == Attribute.NOMINAL){
							int va = (int) Denormalize(Parameters.instanceTRA[k][j],j);
							newInstance += Attributes.getInputAttribute(j).getNominalValue(va);
						}
					}
					
					else{
						if(Parameters.attributeType[j] == Attribute.REAL)
							newInstance +=  Parameters.IS_tra.getInstance(k).getInputRealValues(j);
						if(Parameters.attributeType[j] == Attribute.INTEGER)
							newInstance += (int) Parameters.IS_tra.getInstance(k).getInputRealValues(j);
						if(Parameters.attributeType[j] == Attribute.NOMINAL)
							newInstance +=  Parameters.IS_tra.getInstance(k).getInputNominalValues(j);
					}
					
					newInstance += ", "; 
				}
				
				String className = Parameters.IS_tra.getInstance(k).getOutputNominalValues(0);
				newInstance += className + "\n";
				
				fm.writeLine(newInstance);
			}
				
			fm.closeWrite();
			
		}catch(Exception e){
			e.printStackTrace();
			System.exit(1);
		}
		
	}
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public void createTestFile(){
		// to create the test file-----------------------------------------
		try {
			String s;
			File Archi1 = new File(Parameters.testInputFile);
		    File Archi2 = new File(Parameters.testOutputFile);
		    BufferedReader in;
			in = new BufferedReader(new FileReader(Archi1));
		    PrintWriter out = new PrintWriter(new FileWriter(Archi2));
		      
		    while ((s = in.readLine()) != null)
		    	out.println(s);
		    
		    in.close();
		    out.close();
		}catch (Exception e){
			e.printStackTrace();
		}
	}
	
	
	
/******************************************************************************************************************************************
 * ****************************************************************************************************************************************
 *                            A PARTIR DE AQUI ES LAS FUNCIONES QUE TIENES QUE VER - ISAAC
 * ****************************************************************************************************************************************
 ******************************************************************************************************************************************/

	
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************

	public void run() throws Exception{
		
		FileManagement fm = new FileManagement();
		fm.initWrite(Parameters.logOutputFile);
		
		int numIterations = 0;
		int TOTAL_ITERATIONS = 0;
				
		Initialize();
		if(Parameters.base.equals("1NN")){
			Parameters.ONN = new OptimizedNN();
			current_acc = Parameters.ONN.getInitialAccuracy();
		}
		
		if(Parameters.base.equals("C45")){
			InstanceSet trac45 = createIS();
			C45 c45 = new C45(trac45, trac45);
			current_acc = c45.getAccuracy();
		}
			
			
		
		
		fm.writeLine("==> Initial Fitness = " + current_acc + ","  + "\n\n");
		
		
		do{
			
			TOTAL_ITERATIONS++;
			
			double previous_acc = current_acc;
			
			MoveExamples();
						
			fm.writeLine("Current Fitness at (" + numIterations + ", " + TOTAL_ITERATIONS + ") = " + current_acc + "," + "\n\n");
			
			if(current_acc <= previous_acc){
				numIterations++;
			}
			
			else{
				numIterations = 0;
			}
			
		}while( ((numIterations < Parameters.maxIterations) && (TOTAL_ITERATIONS < 300)) );
		
		CreateTrainingFile();
		createTestFile();
		
		fm.writeLine("\n==> Final Fitness = " + current_acc + "\n");		
		fm.closeWrite();
	}

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
				
	public void Initialize(){
		
		Sampling samp = new Sampling(Parameters.numInstancesTRA);
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			
			int example = samp.getSample();
			
			for(int a = 0 ; a < Parameters.numAttributes ; ++a){
				if(Parameters.missingTRA[example][a]){
					Parameters.instanceTRA[example][a] = ImputeValue_Random(example,a);
				}
			}
		}
		
	}
					
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
		
	public double ImputeValue_Random(int ins, int att){
		
		int classIns = Parameters.classTRA[ins];
		
		double r1 = getValueAtt(classIns, att);
		double r2 = getValueAtt(classIns, att);
		double r3 = getValueAtt(classIns, att);
		
		double res = r1 + Parameters.ScalingFactor*(r2-r3);
		
		if(res < 0)
			res = 0;
		if(res > 1)
			res = 1;

		return res;
	}
		
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
		
	public double getValueAtt(int cla, int att){
		
		int index;
	
		while(Parameters.missingTRA[(index = getInstClass(cla))][att]);
		
		return Parameters.instanceTRA[index][att];			
	}
			
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
		
	public int getInstClass(int cla){

		int val = Randomize.Randint(0, indexCla[cla].size());
		int res = (Integer) indexCla[cla].get(val);

		return res;
	}

//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
		
	public void MoveExamples() throws Exception{
		
		Sampling samp = new Sampling(Parameters.numInstancesTRA);
		
		for(int i = 0 ; i < Parameters.numInstancesTRA ; ++i){
			
			int example = samp.getSample();
			
			for(int a = 0 ; a < Parameters.numAttributes ; ++a){
				
				if(Parameters.missingTRA[example][a]){
					
					double imputation = ImputeValue_Current2Rand(example,a);
					
					int original_nom = -1;
					if(Parameters.attributeType[a] == Attribute.NOMINAL){
						original_nom = Parameters.nominalIntTRA[example][a];
						Parameters.nominalIntTRA[example][a] = (int) Denormalize(imputation, a);
					}
					
					double original_int = -1;
					if(Parameters.attributeType[a] == Attribute.INTEGER){
						original_int = Parameters.integerValueTRA[example][a];
						Parameters.integerValueTRA[example][a] = Normalize(Denormalize(imputation, a), a);
					}
					
					//compruebo que valor funciona mejor si value o si res
					double original_real = Parameters.instanceTRA[example][a];
					Parameters.instanceTRA[example][a] = imputation;
					
					double despues = -1;
					if(Parameters.base.equals("1NN")){
						despues = Parameters.ONN.getOptimizedAccuracy(example, current_acc);
					}
					if(Parameters.base.equals("C45")){
						InstanceSet trac45 = createIS();
						C45 c45 = new C45(trac45, trac45);
						despues = c45.getAccuracy();
					}
					
					
					
					if(despues >= current_acc){
						current_acc = despues;
					}
					else{
						Parameters.instanceTRA[example][a] = original_real;
						if(Parameters.attributeType[a] == Attribute.NOMINAL)
							Parameters.nominalIntTRA[example][a] = original_nom;
						if(Parameters.attributeType[a] == Attribute.INTEGER)
							Parameters.integerValueTRA[example][a] = original_int;
					}
					
				}
			}
		}
	}
	
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
	
	public InstanceSet createIS(){
		
		for(int k = 0 ; k < Parameters.numInstancesTRA ; k++){
			for(int j = 0 ; j < Parameters.numAttributes ; j++){
				
				if(Parameters.missingTRA[k][j]){
					
					if(Parameters.attributeType[j] == Attribute.REAL)
						Parameters.IS_tra_C45.getInstance(k).setInputNumericValue(j, Denormalize(Parameters.instanceTRA[k][j],j));
					if(Parameters.attributeType[j] == Attribute.INTEGER)
						Parameters.IS_tra_C45.getInstance(k).setInputNumericValue(j, (int) Denormalize(Parameters.instanceTRA[k][j],j));
					if(Parameters.attributeType[j] == Attribute.NOMINAL){
						int va = (int) Denormalize(Parameters.instanceTRA[k][j],j);
						Parameters.IS_tra_C45.getInstance(k).setInputNominalValue(j, Attributes.getInputAttribute(j).getNominalValue(va));
					}
					
				}
			
			}
		}
		
		return Parameters.IS_tra_C45;
	}
		
//*****************************************************************************************************************
//*****************************************************************************************************************
//*****************************************************************************************************************
		
	public double ImputeValue_Current2Rand(int ins, int att) throws Exception{
		
		int classIns = Parameters.classTRA[ins];
		double value = Parameters.instanceTRA[ins][att];
		
		double r1 = getValueAtt(classIns, att);
		double r2 = getValueAtt(classIns, att);
		double r3 = getValueAtt(classIns, att);
		
		double res = value + Randomize.RandClosed()*(r1-value) + Parameters.ScalingFactor*(r2-r3);
		
		if(res < 0)
			res = 0;
		if(res > 1)
			res = 1;
		
		return res;
	}	

}