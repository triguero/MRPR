package org.apache.mahout.keel.Algorithms.Preprocess.Missing_Values.MVI_DE;

public class Main {
	
	static public void main(String args[]) throws Exception{
	
		Parameters.doParse(args[0]);
		
		Crono c = new Crono();
		
		c.inicializa();
			MVI_DE method = new MVI_DE();
			method.run();
		c.fin();
		
		System.out.println(c.tiempoTotal());
	}

}