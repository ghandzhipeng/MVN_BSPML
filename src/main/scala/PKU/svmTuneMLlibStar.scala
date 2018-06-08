import org.apache.spark.mllib.classification.{GhandSVMSGDShuffleModelStand, GhandSVMSGDShuffleModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkEnv

object svmTuneMLlibStar {
  def main(args: Array[String]): Unit = {
    val in_path = args(0)
    val num_workers = args(1).toInt
    val reg_para = args(2).toDouble
    val triers = args(3).toInt
    val num_features = args(4).toInt

    val sparkConf = new SparkConf().setAppName("tune-mllib-star")
    val sparkContext = new SparkContext(sparkConf)

    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .repartition(num_workers).persist(StorageLevel.MEMORY_AND_DISK)
    rdd_train_data.setName("cached data")


    val useScaling = SparkEnv.get.conf.get("spark.ml.useFeatureScaling", "true").toBoolean

    var cnt = 0
    while(cnt < triers) {
      val lr = math.pow(10, 1 - cnt)
      if (useScaling) {
        val model = GhandSVMSGDShuffleModelStand.train(input = rdd_train_data,
          numIterations = 1,
          stepSize = lr,
          regParam = reg_para,
          miniBatchFraction = 1)
      }
      else {
        val model = GhandSVMSGDShuffleModel.train(input = rdd_train_data,
          numIterations = 1,
          stepSize = lr,
          regParam = reg_para,
          miniBatchFraction = 1)
      }
    }
    cnt += 1

  }

}
