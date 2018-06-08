import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

object lrWithLBFGS {
  def main(args: Array[String]): Unit = {
    val in_path = args(0)
    val num_iteration = args(1).toInt
    val step_size = args(2).toDouble
    val num_workers = args(3).toInt
    val cores_per_executor = args(4).toInt
    val partition_per_core = args(5).toInt
    val num_features = args(6).toInt
    val reg_para = args(7).toDouble

    val sparkConf = new SparkConf().setAppName("LRwithLBFGS")
    val sparkContext = new SparkContext(sparkConf)

    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .repartition(num_workers * cores_per_executor * partition_per_core).persist(StorageLevel.MEMORY_AND_DISK)

    val model = LogisticRegressionWithLBFGS.train(input = rdd_train_data,
      numIterations = num_iteration,
      regParam = reg_para, numCorrections = 10)

  }

}
