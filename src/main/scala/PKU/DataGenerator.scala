import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.util.Random

object DataGenerator{
  def main(args: Array[String]): Unit= {
    val num_instance = args(0).toInt // instance per task
    val num_feature = args(1).toInt
    val sparsity = args(2).toDouble
    val positive_ratio = args(3).toDouble // how many postive instances
    val num_task = args(4).toInt
    val output_path = args(5).toString

    val sparkConf = new SparkConf().setAppName("dataGenerator")
    val sparkContext = new SparkContext(sparkConf)

    val array: Array[Int] = new Array[Int](num_task)
    val mu = sparsity * num_feature
    val sigma = 1.0


    val tasks: RDD[Int] = sparkContext.parallelize(array, num_task)
    val generatedData: RDD[String] = tasks.map {
      _ =>
        val stringBuffer: StringBuffer = new StringBuffer()

        val nnzPerInstanceGen = new Random() // generate the nnz of features per instance
        val labelGen = new Random() // generate the label of each instance
        val featureSelectGen = new Random() // to decide whether one dimension of a instance is zero.
        val featureGen = new Random() // generate the value of features

        for (i <- 0 to num_instance) {
          // decide the label of this instance
          if (labelGen.nextFloat() < positive_ratio)
            stringBuffer.append(1)
          else
            stringBuffer.append(0)

          // generate the feature and values
          val nnz: Int = Math.max((nnzPerInstanceGen.nextGaussian() + mu).toInt, 1)
          val possibility: Float = nnz / num_feature.toFloat
          for (j <- 0 to num_feature) {
            if (featureSelectGen.nextFloat() < possibility) {
              stringBuffer.append(" " + j + ":" + featureGen.nextFloat())
            }
          }
          stringBuffer.append("\n")
        }

        stringBuffer.toString

    }
//    println(generatedData.count())
    generatedData.saveAsTextFile("fakeData/" + output_path)
  }
}
