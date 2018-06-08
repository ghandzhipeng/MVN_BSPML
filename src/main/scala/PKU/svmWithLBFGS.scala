import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{LabeledPoint => ml_LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, SparkEnv}


object svmWithLBFGS {
  def main(args: Array[String]): Unit = {
    val in_path = args(0)
    val num_iteration = args(1).toInt
    val budget = args(2).toInt
    val step_size = args(3).toDouble
    val num_workers = args(4).toInt
    val cores_per_executor = args(5).toInt
    val partition_per_core = args(6).toInt
    val reg_para = args(7).toDouble
    val num_features = args(8).toInt


    val sparkConf = new SparkConf().setAppName("LinearSVC")
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sparkContext)

    // convert mllib labeledPoint to ml labeledpoint
    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .map(x => ml_LabeledPoint(x.label, x.features.asML))
    rdd_train_data.setName("cached data")

    val train_data = sqlContext.createDataFrame(rdd_train_data)
      .repartition(num_workers * cores_per_executor * partition_per_core)
      .persist(StorageLevel.MEMORY_AND_DISK)

    //    val num_features: Int = rdd_data.map(_.features.size).first()

    val lsvc = new LinearSVC()
      .setMaxIter(num_iteration)
      .setRegParam(reg_para)
      .setFitIntercept(false)
      .setStandardization(false)
      .setTol(0)

    lsvc.setMaxIter(num_iteration)
    lsvc.SetBudget(budget)

    val lsvcModel = lsvc.fit(train_data)

  }

}
