import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.classification.GhandLogRSGDShuffleModel

object lrWithSGDws{
  def main(args: Array[String]): Unit={
    val in_path = args(0)
    val num_iteration = args(1).toInt
    val mini_batch_fraction = args(2).toDouble
    val step_size = args(3).toDouble
    val num_workers = args(4).toInt
    val cores_per_executor = args(5).toInt
    val partition_per_core = args(6).toInt
    val reg_para = args(7).toDouble

    val sparkConf = new SparkConf().setAppName("lrWithSGD")
    val sparkContext = new SparkContext(sparkConf)

    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path)
        .repartition(num_workers * cores_per_executor * partition_per_core)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val model = GhandLogRSGDShuffleModel.train(input = rdd_train_data,
      numIterations = num_iteration,
      stepSize = step_size,
      regParam = reg_para,
      miniBatchFraction = mini_batch_fraction
    )

  }

}

