package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UndersamplingReducer_v2 extends Reducer<LongWritable, Text, NullWritable, Text>{
	
  private static final Logger log = LoggerFactory.getLogger(UndersamplingReducer_v2.class);
	  
  public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
    NullWritable id = null;

    for (Text value : values) {  
      context.write(id, value);
  	}  
  }

}
