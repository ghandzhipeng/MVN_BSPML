import org.apache.spark.mllib.classification.{GhandSVMSGDShuffleModel, GhandSVMSGDShuffleModelStand, SVMWithSGD}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkEnv

object svmTuneMLlib {
  def main(args: Array[String]): Unit = {
    val in_path = args(0)
    val num_workers = args(1).toInt
    val reg_para = args(2).toDouble
    val triers = args(3).toInt
    val num_features = args(4).toInt

    val sparkConf = new SparkConf().setAppName("tune-mllib")
    val sparkContext = new SparkContext(sparkConf)
    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .repartition(num_workers).persist(StorageLevel.MEMORY_AND_DISK)
    rdd_train_data.setName("cached data")

    var cnt = 0
    while (cnt < triers) {
      val lr = math.pow(10, 1 - cnt)
      val model = SVMWithSGD.train(input = rdd_train_data,
        numIterations = 10,
        stepSize = lr,
        regParam = reg_para,
        miniBatchFraction = 0.01)
      cnt += 1
    }

  }

}
