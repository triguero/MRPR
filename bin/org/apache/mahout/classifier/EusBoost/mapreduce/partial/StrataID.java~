/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.smo.mapreduce.partial;

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.LongWritable;

/**
 * Indicates both the strata and the data partition used to grow the strata
 */
public class StrataID extends LongWritable implements Cloneable {
  
  public static final int MAX_TREEID = 100000;
  
  public StrataID() { }
  
  public StrataID(int partition, int strataId) {
    Preconditions.checkArgument(partition >= 0, "partition < 0");
    Preconditions.checkArgument(strataId >= 0, "strataId < 0");
    set(partition, strataId);
  }
  
  public void set(int partition, int strataId) {
    set((long) partition * MAX_TREEID + strataId);
  }
  
  /**
   * Data partition (InputSplit's index) that was used as the strata
   */
  public int partition() {
    return (int) (get() / MAX_TREEID);
  }
  
  public int strataId() {
    return (int) (get() % MAX_TREEID);
  }
  
  @Override
  public StrataID clone() {
    return new StrataID(partition(), strataId());
  }
}
