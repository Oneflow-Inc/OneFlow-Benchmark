[TOC]

[HugeCTR](https://github.com/NVIDIA/HugeCTR)是英伟达提供的一种高效的GPU框架，专为点击率（CTR）估计训练而设计。

OneFlow对标HugeCTR搭建了Wide & Deep 网络，本文介绍如何为该网络准备数据集。

Wide and Deep Learning (WDL)

另1：本文作者没有太多spark和scala的相关经验。由于绝大多数数据集的处理方式是列操作，导致内存溢出频繁发生。本来想提供一个完整的工具，比如一个可执行的jar包，后来还是在Spark环境下交互式操作完成的这项工作。本文把交互的步骤罗列下来，中间有一些冗余的代码也没有处理，待后面整理。
另2：整个处理过程最大内存消耗是170G，建议给Spark开192G以上的内存。

## 数据集及预处理
数据由[CriteoLabs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)提供。原始数据包括三个部分：一个标签列`labels`、13个整型特征`I列`、26个分类特征`C列`。数据处理后：
- `I列`转换为`dense_fields`；
- `C列`转换为`deep_sparse_fields`;
- `C列`中的`C1 C2`、`C3 C4`构成了交叉特征，形成了`wide_sparse_fields`。

数据经过处理后保存成`ofrecord`格式，结构如下：
```
root
 |-- deep_sparse_fields: array (nullable = true)
 |    |-- element: integer (containsNull = true)
 |-- dense_fields: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- labels: integer (nullable = true)
 |-- wide_sparse_fields: array (nullable = true)
 |    |-- element: integer (containsNull = true)

```
## step0 准备工作
这一步主要是导入相关的库，并且准备一个临时目录。后面的很多步骤中都主动把中间结果保存到临时目录中，这样能够节省内存，而且方便中断恢复操作。
```
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{when, _}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, MinMaxScaler}
import org.apache.spark.ml.linalg._

import java.nio.file.{Files, Paths}
val tmp_dir = "/DATA/disk1/xuan/wdl_tmp"
Files.createDirectories(Paths.get(tmp_dir))
```
## step1 导入数据
这一步中读入原始数据集，并根据需求做了如下操作：
1. 给读入的每一列命名[label, I1,...,I13, C1,...,C26]
2. 给每一条数据加上`id`，后面所有的表的合并操作都基于这个`id`
3. 将`I列`转换成整型
4. `I列`和`C列`空白处补`NaN`
5. `features`是后面经常用到的DataFrame
```
// load input file
var input = spark.read.options(Map("delimiter"->"\t")).csv("file:///DATA/disk1/xuan/train.shuf.bak")
// var input = spark.read.options(Map("delimiter"->"\t")).csv("file:///DATA/disk1/xuan/train.shuf.txt")

// rename columns [label, I1,...,I13, C1,...,C26]
val NUM_INTEGER_COLUMNS = 13
val NUM_CATEGORICAL_COLUMNS = 26

// val integer_cols = (1 to NUM_INTEGER_COLUMNS).map{id=>s"I$id"}
val integer_cols = (1 to NUM_INTEGER_COLUMNS).map{id=>$"I$id"} // note
val categorical_cols = (1 to NUM_CATEGORICAL_COLUMNS).map{id=>s"C$id"}
val feature_cols = integer_cols.map{c=>c.toString} ++ categorical_cols
val all_cols = (Seq(s"labels") ++ feature_cols)
input = input.toDF(all_cols: _*).withColumn("id", monotonically_increasing_id())

input = input.withColumn("labels", col("labels").cast(IntegerType))
// cast integer columns to int
for(i <- 1 to NUM_INTEGER_COLUMNS) {
  val col_name = s"I$i"
  input = input.withColumn(col_name, col(col_name).cast(IntegerType))
}

// replace `null` with `NaN`
val features = input.na.fill(Int.MinValue, integer_cols.map{c=>c.toString}).na.fill("80000000", categorical_cols)

// dump features as parquet format
val features_dir = tmp_dir ++ "/filled_features"
features.write.mode("overwrite").parquet(features_dir)
```
Mem: 52.6G
duration: 1 min
## step2 处理整型特征生成`dense_fields`
需要两个步骤：
1. 循环处理每一个`I列`，编码映射后保存到临时文件夹；
2. 从临时文件夹中读取后转换成`dense_fields`并保存。
### `I列`编码映射
对于每一个整型特征：
- 计算每个特征值的频次
- 频次小于6的特征值修改为NaN
- 特征编码
- 进行normalize操作，或仅+1操作
- 保存该列到临时文件夹
```
val features_dir = tmp_dir ++ "/filled_features"
val features = spark.read.parquet(features_dir)

// integer features
println("create integer feature cols")
val normalize_dense = 1
val nanValue = Int.MinValue
val getItem = udf((v: Vector, i: Int) => v(i).toFloat)
for(column_name <- integer_cols) {
  val col_name = column_name.toString
  println(col_name)
  val col_index = col_name ++ "_index"
  val uniqueValueCounts = features.groupBy(col_name).count()
  val df = features.join(uniqueValueCounts, Seq(col_name))
                   .withColumn(col_name, when(col("count") >= 6, col(col_name)).otherwise(nanValue))
                   .select("id", col_name)
  val indexedDf = new StringIndexer().setInputCol(col_name)
                                     .setOutputCol(col_index)
                                     .fit(df).transform(df)
                                     .drop(col_name) // trick: drop col_name here and will be reused later
  
  var scaledDf = spark.emptyDataFrame
  if (normalize_dense > 0) {
      val assembler = new VectorAssembler().setInputCols(Array(col_index)).setOutputCol("vVec")
      val df= assembler.transform(indexedDf)
      scaledDf = new MinMaxScaler().setInputCol("vVec")
                                   .setOutputCol(col_name)
                                   .fit(df).transform(df)
                                   .select("id", col_name)
  } else {
      scaledDf = indexedDf.withColumn(col_name, col(col_index) + lit(1)) // trick: reuse col_name
                          .select("id", col_name)
                          //.withColumn(col_name, col(col_index).cast(IntegerType))
  }
  val col_dir = tmp_dir ++ "/" ++ col_name
  scaledDf = scaledDf.withColumn(col_name, getItem(column_name, lit(0)))
  scaledDf.write.mode("overwrite").parquet(col_dir)
  scaledDf.printSchema
}
```
Mem: 58.6G 
duration: 3*13 ~= 40 min
### 合并所有`I列`形成`dense_fields`
- 从临时文件夹里分别读取各列，并合并到一个dataframe `df`里；
- 将`df`里的`I列`合并成`dense_fields`;
- 将`dense_fields`保存到临时文件夹。
```
val integer_cols = (1 to NUM_INTEGER_COLUMNS).map{id=>s"I$id"} 
var df = features.select("id")
for(col_name <- integer_cols) {
  println(col_name)
  val df_col = spark.read.parquet(tmp_dir ++ "/" ++ col_name)
  df = df.join(df_col, Seq("id"))
}
df = df.select($"id", array(integer_cols map col: _*).as("dense_fields"))
val parquet_dir = tmp_dir ++ "/parquet_dense"
df.write.mode("overwrite").parquet(parquet_dir)
```
Mem: 110G
Duration: 3mins
## step3 处理分类特征和交叉特征并生成`deep_sparse_fields`和`wide_sparse_fields`
### 处理分类特征
对于每一个分类特征：
- 计算每个特征值的频次
- 频次小于6的特征值修改为NaN
- 特征编码
- 编码后的值加offset
- 保存该列到临时文件夹

需要注意的是，offset的初始值设为1，而且offset是随着`C列`递增的，而且最后的offset还要继续被用到交叉特征里。
```
println("create categorical feature cols")
val nanValue = "80000000"
var offset: Long = 1
for(col_name <- categorical_cols) {
  println(col_name)
  val col_index = col_name ++ "_index"
  val uniqueValueCounts = features.groupBy(col_name).count()
  val df = features.join(uniqueValueCounts, Seq(col_name))
                   .withColumn(col_name, when(col("count") >= 6, col(col_name)).otherwise(nanValue))
                   .select("id", col_name)
  val indexedDf = new StringIndexer().setInputCol(col_name)
                                     .setOutputCol(col_index)
                                     .fit(df).transform(df).drop(col_name)
                                     
  val scaledDf = indexedDf.withColumn(col_name, col(col_index) + lit(offset))
                          .withColumn(col_name, col(col_name).cast(IntegerType))
                          .drop(col_index)
  val hfCount = indexedDf.select(col_index).distinct.count()
  println(offset, hfCount)
  offset += hfCount
  val col_dir = tmp_dir ++ "/" ++ col_name
  scaledDf.write.mode("overwrite").parquet(col_dir)
}
```
Mem: 110G
Time: 1 hour
下面是输出结果中代表某个`C列`的offset和独立值个数。下面这段不是程序脚本，是输出结果。
```
C1
(1,1460)
C2
(1461,558)
C3
(2019,335378)
C4
(337397,211710)
C5
(549107,305)
C6
(549412,20)
C7
(549432,12136)
C8
(561568,633)
C9
(562201,3)
C10
(562204,51298)
C11
(613502,5302)
C12
(618804,332600)
C13
(951404,3179)
C14
(954583,27)
C15
(954610,12191)
C16
(966801,301211)
C17
(1268012,10)
C18
(1268022,4841)
C19
(1272863,2086)
C20
(1274949,4)
C21
(1274953,324273)
C22
(1599226,17)
C23
(1599243,15)
C24
(1599258,79734)
C25
(1678992,96)
C26
(1679088,58622)
```
### 生成交叉特征
首先要注意：这一步依赖前面offset的值。
- 交叉特征是由两个分类特征组合而成，组合方式是：col0 * col1_width + col1。
- 组合后的值进行编码
- 编码后保存到临时文件夹
```
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, MinMaxScaler}
var offset: Long = 1737710
println("create cross cols")
val feature_pairs = Array(Array("C1", "C2"), Array("C3", "C4"))
for(feature_pair <- feature_pairs) {
  val col0 = feature_pair(0)
  val col1 = feature_pair(1)  
  val df_col0 = spark.read.parquet(tmp_dir ++ "/" ++ col0)
  val df_col1 = spark.read.parquet(tmp_dir ++ "/" ++ col1)
  
  val cross_col = col0 ++ "_" ++ col1
  val cross_col_index = cross_col ++ "_index"
  println(cross_col)
  
  val min_max = df_col1.agg(min(col1), max(col1)).head()
  val col1_width = min_max.getInt(1) - min_max.getInt(0) + 1
  val df = df_col0.withColumn(col0, col(col0) * lit(col1_width))
                  .join(df_col1, Seq("id"))
                  .withColumn(cross_col, col(col0) + col(col1))
                  .select("id", cross_col)
  val indexedDf = new StringIndexer().setInputCol(cross_col)
                                     .setOutputCol(cross_col_index)
                                     .fit(df).transform(df).drop(cross_col)
                                     
  val scaledDf = indexedDf.withColumn(cross_col, col(cross_col_index).cast(IntegerType))
                          .withColumn(cross_col, col(cross_col) + lit(offset))
                          .drop(cross_col_index)  
                          .withColumn(cross_col, col(cross_col).cast(IntegerType))
                          
  val hfCount = indexedDf.select(cross_col_index).distinct.count()
  println(offset, hfCount)
  offset += hfCount
  val col_dir = tmp_dir ++ "/" ++ cross_col
  scaledDf.write.mode("overwrite").parquet(col_dir)
  scaledDf.show
  scaledDf.printSchema
}
```
Mem: 110G
Duration: 2mins
```
(1737710,144108)
(1881818,447097)
```
### 生成`deep_sparse_fields`
这段操作和形成`dense_fields`的方式相似，代码冗余。
这一段要处理26个列，内存消耗极大（170G），速度到不是最慢的。如果数据集更大，或可采用每次合一列的方式。前面的`dense_fields`也可以采用这种方式，列为`TODO`吧。
```
val tmp_dir = "/DATA/disk1/xuan/wdl_tmp"
val features_dir = tmp_dir ++ "/filled_features"
val features = spark.read.parquet(features_dir)

val NUM_CATEGORICAL_COLUMNS = 26
val categorical_cols = (1 to NUM_CATEGORICAL_COLUMNS).map{id=>s"C$id"}

var df = features.select("id")
for(col_name <- categorical_cols) {
  println(col_name)
  val df_col = spark.read.parquet(tmp_dir ++ "/" ++ col_name)
  df = df.join(df_col, Seq("id"))
}
df = df.select($"id", array(categorical_cols map col: _*).as("deep_sparse_fields"))
val parquet_dir = tmp_dir ++ "/parquet_deep_sparse"
df.write.mode("overwrite").parquet(parquet_dir)
```
Mem: 170G
Duration: 5min

### 生成`wide_sparse_fields`
这段操作和形成`dense_fields`的方式相似，代码冗余。
```
val cross_pairs = Array("C1_C2", "C3_C4")
var df = features.select("id")
for(cross_pair <- cross_pairs) {
  val df_col = spark.read.parquet(tmp_dir ++ "/" ++ cross_pair)
  df = df.join(df_col, Seq("id"))
}
df = df.select($"id", array(cross_pairs map col: _*).as("wide_sparse_fields"))
val parquet_dir = tmp_dir ++ "/parquet_wide_sparse"
df.write.mode("overwrite").parquet(parquet_dir)
```
Duration: 1min
## step4 合并所有字段
```
val fields = Array("dense", "deep_sparse", "wide_sparse")
var df = features.select("id", "labels")
for(field <- fields) {
  val df_col = spark.read.parquet(tmp_dir ++ "/parquet_" ++ field)
  df = df.join(df_col, Seq("id"))
}
val parquet_dir = tmp_dir ++ "/parquet_all"
df.write.mode("overwrite").parquet(parquet_dir)
```
## step5 写入ofrecord
```
val tmp_dir = "/DATA/disk1/xuan/wdl_tmp"
import org.oneflow.spark.functions._
val parquet_dir = tmp_dir ++ "/parquet_all"
val df = spark.read.parquet(parquet_dir)

val dfs = df.drop("id").randomSplit(Array(0.8, 0.1, 0.1))

val ofrecord_dir = tmp_dir ++ "/ofrecord/train"
dfs(0).repartition(256).write.mode("overwrite").ofrecord(ofrecord_dir)
dfs(0).count
sc.formatFilenameAsOneflowStyle(ofrecord_dir)

val ofrecord_dir = tmp_dir ++ "/ofrecord/val"
dfs(1).repartition(256).write.mode("overwrite").ofrecord(ofrecord_dir)
dfs(1).count
sc.formatFilenameAsOneflowStyle(ofrecord_dir)

val ofrecord_dir = tmp_dir ++ "/ofrecord/test"
dfs(2).repartition(256).write.mode("overwrite").ofrecord(ofrecord_dir)
dfs(2).count
sc.formatFilenameAsOneflowStyle(ofrecord_dir)
```


