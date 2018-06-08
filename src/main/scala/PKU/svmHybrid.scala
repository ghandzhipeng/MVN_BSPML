import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{LabeledPoint => ml_LabeledPoint}
import org.apache.spark.mllib.classification.GhandSVMSGDShuffleModel
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, TaskContext}

object svmHybrid extends Logging {
  // Note for this, have to set the featureStd as (1) in LinearSVC. Manaually setting.

  /**
    * @param iter : iterator for enumerate the data
    * @return (loss, learning_rate)
    */
  def tunning_partition(iter: Iterator[LabeledPoint], feature_size: Int, L2Reg: Double): (Double, Double) = {
    // assume the features are sparse and models are dense.
    // step-size = 10 ^ {-partitionId} * 10
    val stepSize = math.pow(10, 1 - TaskContext.getPartitionId())
    if (stepSize * L2Reg > 0.1) {
      // if L2Reg is big, then we need very small step size.
      return (Double.MaxValue, stepSize)
    }

    var loss: Double = 0
    val local_model = OldVectors.zeros(feature_size).toDense
    var factor: Double = 1.0
    val llist = iter.toList
    val iter1 = llist.iterator
    val iter2 = llist.iterator

    while (iter1.hasNext) {
      val dataPoint: LabeledPoint = iter1.next()
      if (factor < 1e-5) {
        MYBLAS.scal(factor, local_model)
        factor = 1.0
      }
      val transStepSize = stepSize / (1 - stepSize * L2Reg) / factor
      val dotProduct = MYBLAS.dot(dataPoint.features, local_model) * factor // because model is not the real model here.
      val labelScaled = 2 * dataPoint.label - 1.0
      val local_loss = if (1.0 > labelScaled * dotProduct) {
        MYBLAS.axpy((-labelScaled) * (-transStepSize), dataPoint.features, local_model)
        1.0 - labelScaled * dotProduct
      } else {
        0
      }
      factor *= (1 - stepSize * L2Reg)
      loss += local_loss
    }

    MYBLAS.scal(factor, local_model) // local_model = local_model * factor
    var real_loss: Double = 0.0
    while (iter2.hasNext) {
      val dataPoint = iter2.next()
      val dotProduct = MYBLAS.dot(dataPoint.features, local_model) * (2 * dataPoint.label - 1)
      if (dotProduct < 1) {
        real_loss += 1 - dotProduct
      }
    }

    logInfo(s"ghand=tunning=loss:${loss}=realLoss:${real_loss}stepSize=${stepSize}")
    (real_loss, stepSize)
  }

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

    val sparkConf = new SparkConf().setAppName("Hybrid-without-feature-standardization")
    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sparkContext)

    val rdd_train_data = MLUtils.loadLibSVMFile(sparkContext, in_path, numFeatures = num_features)
      .repartition(num_workers * cores_per_executor * partition_per_core)
      .persist(StorageLevel.MEMORY_AND_DISK)
    rdd_train_data.setName("cached data")

    val df_data = rdd_train_data.map(x => ml_LabeledPoint(x.label, x.features.asML))
    val df_train_data = sqlContext.createDataFrame(df_data)

    // hyper-parameter tunning for model averaging
    val partiallyReduced = rdd_train_data.mapPartitions {
      it =>
        Iterator(tunning_partition(it, num_features, reg_para))
    }
    val tunning_result: (Double, Double) = partiallyReduced.reduce {
      (x, y) => {

        if (x._1 < y._1) {
          logInfo(s"ghand=tunning=LR:${y._2}=loss:${y._1}")
          x
        }
        else {
          logInfo(s"ghand=tunning=LR:${x._2}=loss:${x._1}")
          y
        }

      }
    }
    val learning_rate = tunning_result._2
    logInfo(s"ghand=tunning=selectedLR:${learning_rate}=loss:${tunning_result._1}")

    // train using model averaging + reduce-scatter
    val model = GhandSVMSGDShuffleModel.train(input = rdd_train_data,
      numIterations = 1,
      stepSize = learning_rate,
      regParam = reg_para,
      miniBatchFraction = 1.0)

    // use model from model averaging as intialGuess
    val lsvc = new LinearSVC()
      .setMaxIter(num_iteration)
      .setRegParam(reg_para)
      .setFitIntercept(false)
      .setStandardization(false)
      .setTol(0)

    lsvc.SetBudget(budget)
    lsvc.setInitialModel(model.weights.toArray)

    val lsvcModel = lsvc.fit(df_train_data)

  }
}
