MRPR
====

This repository includes the MapReduce implementation proposed for Prototype Reduction in [1].
This implementation is based on Apache Mahout 0.8 library. The Apache Mahout (http://mahout.apache.org/) project's goal is to build an environment for quickly creating scalable performant machine learning applications.

Prerequisites:
- Hadoop 2.5.
- ant

Associated papers:

[1] I. Triguero, D. Peralta, J. Bacardit, S. García, F. Herrera. MRPR: A MapReduce Solution for Prototype Reduction in Big Data Classification. Neurocomputing 150 (2015), 331-345. doi: 10.1016/j.neucom.2014.04.078
(http://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1769_2015-Neurocomputing-MRPR-A%20MapReduce%20solution%20for%20prototype%20reduction%20in%20big%20data%20classification.pdf)

[2] I. Triguero, D. Peralta, J. Bacardit, S. García, F. Herrera. A Combined MapReduce-Windowing Two-Level Parallel Scheme for Evolutionary Prototype Generation. In Proceeding on the WCCI 2014 IEEE World Congress on Computational Intelligence, IEEE Congress on Evolutionary Computation CEC'2014, Beijing (China), 6-11 July, pp. 3036-3043, 2014. 


Compile the whole project with ANT:
<pre>
$ ant
</pre>

Put the dataset folder into the HDFS system:
<pre>
hadoop fs -put datasets/
</pre>

Generate descriptor file needed by the mahout code. (Check: ...classifier.df.tools.Describe.java).
<pre>
$ hadoop jar Model.jar org.apache.mahout.classifier.df.tools.Describe -p  datasets/page-blocks-10-fold/page-blocks-10-1tra.data  -f  datasets/page-blocks-10-fold/page-blocks.info -d  10 N L
</pre>


== 
PrototypeGenerationModel class 
==
hadoop jar Model.jar  org.apache.mahout.classifier.pg.mapreduce.PrototypeGenerationModel --help
<pre>
Usage:                                                                          
 [--data <path> --dataset <dataset> --header <header> --output <path> --help    
--pgMethod <path> --TypeOfReduce <path> --numberOfWindows <path>]               
Options                                                                         
  --data (-d) path               Data path                                      
  --dataset (-ds) dataset        The path of the file descriptor of the dataset 
  --header (-he) header          Header of the dataset in Keel format           
  --output (-o) path             Output path, will contain the preprocessed     
                                 dataset                                        
  --help (-h)                    Print out help                                 
  --pgMethod (-pg) path          PG method: IPADE or SSMASFLSDE. Default: IPADE 
  --TypeOfReduce (-r) path       Type of reduce: Join, Fusion, Filtering,       
                                 NoReduce. Default: Join                        
  --numberOfWindows (-w) path    Number of Windows     
</pre>

Generate the Reduced Set example:

To compute the number of mappers, we have to check the number of bytes of the training file:
<pre>
$ ls -l datasets/page-blocks-10-fold/page-blocks-10-1tra.data 
 -rw-rw-r-- 1 isaac isaac 221580 jul 15  2013 datasets/page-blocks-10-fold/page-blocks-10-1tra.data 
</pre>

If we want to have 4 maps, we should divide this number by 4 (55395).

<pre>
hadoop jar Model.jar  org.apache.mahout.classifier.pg.mapreduce.PrototypeGenerationModel -Dmapred.min.split.size=55395 -Dmapred.max.split.size=55396   -d datasets/page-blocks-5-fold/page-blocks-5-1tra.data  -he datasets/page-blocks-5-fold/page-blocks.header  -ds datasets/page-blocks-5-fold/page-blocks.info -pg SSMASFLDE -r Fusion -o output-MRPR/
</pre>


==
TestModel class 
==

<pre>
hadoop jar Model.jar  org.apache.mahout.classifier.pg.mapreduce.TestModel --help


Usage:                                                                          
 [--input <input> --info <test> --header <header> --preprocessed <path> --save  
Reduce set as plain text <path> --output <output> --help]         

Options                                                                         
  --input (-i) input                              Path to job input directory.  
  --info (-ds) test                               The path of the file          
                                                  descriptor of the dataset     
  --header (-he) header                           Header of the dataset in Keel 
                                                  format                        
  --preprocessed (-pre) path                      Preprocessed set path         
  --save Reduce set as plain text (-save) path    Preprocessed set path         
  --output (-o) output                            The directory pathname for    
                                                  output.                       
  --help (-h)                                     Print out help      
</pre>

Classifying the test set example:

Number of mappers of the test phase. Bytes of the test file: 24706. We will divide this into two parts:

<pre>
hadoop jar Model.jar  org.apache.mahout.classifier.pg.mapreduce.TestModel -Dmapred.min.split.size=12353 -Dmapred.max.split.size=12354 -i datasets/page-blocks-10-fold/page-blocks-10-1tst.data 
-ds datasets/page-blocks-10-fold/page-blocks.info -he datasets/page-blocks-5-fold/page-blocks.header --pre output-MRPR/resultingSet.data -o outputKNN-pageblocks
</pre>
