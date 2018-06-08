import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.mllibFP.util.MLUtils
import org.apache.spark.mllibFP.classfication.FeatureParallelMaster

object FP{
  def main(args: Array[String]): Unit ={
    val in_path = args(0)
    val num_partitions = args(1).toInt
    val num_features = args(2).toInt

    val model = SparkEnv.get.conf.get("spark.ml.model", "SVM").toUpperCase
    val step_size = args(3).toDouble
    val mini_batch_size = args(4).toInt
    val reg_para = args(5).toDouble
    val num_iteration = args(6).toInt

    val sparkConf = new SparkConf().setAppName("FP-" + model)
    val sparkContext = new SparkContext(sparkConf)

    val fp_rdd = MLUtils.loadLibSVMFileFeatureParallel(sparkContext, in_path, num_features, num_partitions)

    FeatureParallelMaster.trainMiniBatchSGD(input = fp_rdd._1, labels = fp_rdd._2,
      numFeatures = num_features,
      numPartitions = num_partitions,
      regParam = reg_para,
      stepSize = step_size,
      numIterations = num_iteration,
      miniBatchSize = mini_batch_size,
      modelName = model
    )
  }
}