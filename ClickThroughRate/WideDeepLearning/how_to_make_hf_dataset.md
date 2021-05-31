# How to Make High Frequency Dataset for OneFlow-WDL

[**how_to_make_ofrecord_for_wdl**](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md)一文中介绍了如何利用spark制作OneFlow-WDL使用的ofrecord数据集，GPU&CPU混合embedding的实践中，这个数据集就不好用了，主要原因是没有按照词频排序，所以需要制作新的数据集。本文将持续上文中的套路，介绍一下如何制作按照词频排序的数据集。

## 数据集及预处理

数据由[CriteoLabs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)提供。原始数据包括三个部分：一个标签列`labels`、13个整型特征`I列`、26个分类特征`C列`。数据处理后：

- `I列`转换为`dense_fields`；
- `C列`转换为`deep_sparse_fields`;
- `C列`中的`C1 C2`、`C3 C4`构成了交叉特征，形成了`wide_sparse_fields`。

数据经过处理后保存成`ofrecord`格式，结构如下：

```bash
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

```scala
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

```scala
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

```scala
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

duration: 3*13 ~= 40 min

### 合并所有`I列`形成`dense_fields`

- 从临时文件夹里分别读取各列，并合并到一个dataframe `df`里；
- 将`df`里的`I列`合并成`dense_fields`;
- 将`dense_fields`保存到临时文件夹。

```scala
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

Duration: 3mins

## step3 处理分类特征并生成`deep_sparse_fields`

在这一步中，我们首先处理`deep_sparse_fields`，也就是`C*`列数据。因为在原来的处理中，所有的id都需要加上一个offset，这样就保证了所有的C列id不会重复，但也导致了后面列高频词的id要比前面列低频词的id还要大，所以无法满足需求。为了解决这个问题，需要拿到所有的C列，得到不重复的词的列表，并且按照词频排序，为了做到列之间即使出现相同的词也能有不同的id，我们会在每列词的前面加上列名作为前缀，然后再计算词频并排序。下面介绍具体过程：

#### 处理分类特征 

- 创建`new_categorical_cols`用于给所有的分类特征值加上列的名称
- 选择新的分类特征列，并保存到spark session中，表名为`f`
- 获得表`f`中的所有列的所有不重复的值，并且按照频次从高到低排序，结果存到uniqueValueCounts里面，忽略频次小于6的值
- 按照频次的高低给每一个值分配一个`fid`，即频次最高的为0，存到`hf`表中
- 再重新遍历所有的分类表，用新的`fid`替换原来的特征，并保存到文件系统

```scala
val new_categorical_cols = (1 to NUM_CATEGORICAL_COLUMNS).map{id=>concat(lit(s"C$id"), col(s"C$id")) as s"C$id"}
features.select(new_categorical_cols:_*).createOrReplaceTempView("f")
val orderedValues = spark.sql("select cid, count(*) as cnt from (select explode( array(" + categorical_cols.mkString(",") + ") ) as cid  from f) group by cid ").filter("cnt>=6").orderBy($"cnt".desc)

val hf = orderedValues.select("cid").as[(String)].rdd.zipWithIndex().toDF().select(col("_1").as("cid"), col("_2").as("fid"))

for(col_name <- categorical_cols) {
  println(col_name)
  val col_feature = features.select(col("id"), concat(lit(col_name), col(col_name)) as col_name)
  val scaledDf = col_feature.join(hf, col_feature(col_name)=== hf("cid")).select(col("id"), col("fid").as(col_name))
  val col_dir = tmp_dir ++ "/" ++ col_name
  scaledDf.write.mode("overwrite").parquet(col_dir)
}
```

Mem: 110G
Time: 10 mins

### 生成`deep_sparse_fields`

这段操作和形成`dense_fields`的方式相似，代码冗余。
这一段要处理26个列，内存消耗极大（170G），速度到不是最慢的。如果数据集更大，或可采用每次合一列的方式。前面的`dense_fields`也可以采用这种方式，列为`TODO`吧。

```scala
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

Duration: 5min

## Step4 生成交叉特征并生成wide_sparse_fields

在OneFlow-WDL里，交叉特征被用来生成`wide_sparse_fields`，也是有可能需要按照高低频排序的。在之前交叉特征的id被排在了后面，存在将交叉特征和分类特征一起使用的可能，即使用同一个embedding表。如果这里单独按照高低频排序，就不能这么做了，不过不影响当前的WDL网络。

```scala
val cross_pairs = Array("C1_C2", "C3_C4")
var df = features.select("id")
for(cross_pair <- cross_pairs) {
  val df_col = spark.read.parquet(tmp_dir ++ "/" ++ cross_pair)
  df = df.join(df_col, Seq("id"))
}
// df.select("C1_C2", "C3_C4").createOrReplaceTempView("f")
// df.select(cross_pairs.map{id=>col(id)}:_*).createOrReplaceTempView("f")
df.select(cross_pairs map col: _*).createOrReplaceTempView("f")

val orderedValues = spark.sql("select cid, count(*) as cnt from (select explode( array(" + cross_pairs.mkString(",") + ") ) as cid  from f) group by cid ").filter("cnt>=6").orderBy($"cnt".desc)

val hf = orderedValues.select("cid").as[(String)].rdd.zipWithIndex().toDF().select(col("_1").as("cid"), col("_2").as("fid"))

for(cross_pair <- cross_pairs) {
  df = df.join(hf, df(cross_pair)=== hf("cid")).drop(cross_pair, "cid").withColumnRenamed("fid", cross_pair)
}

df = df.select($"id", array(cross_pairs map col: _*).as("wide_sparse_fields"))
val parquet_dir = tmp_dir ++ "/parquet_wide_sparse"
df.write.mode("overwrite").parquet(parquet_dir)
```

Duration: 2min

## step5 合并所有字段

```scala
val fields = Array("dense", "deep_sparse", "wide_sparse")
var df = features.select("id", "labels")
for(field <- fields) {
  val df_col = spark.read.parquet(tmp_dir ++ "/parquet_" ++ field)
  df = df.join(df_col, Seq("id"))
}
val parquet_dir = tmp_dir ++ "/parquet_all"
df.write.mode("overwrite").parquet(parquet_dir)
```

## Step6 写入ofrecord

```scala
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

