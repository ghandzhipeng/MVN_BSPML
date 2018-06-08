import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}

object partition_data{
  def main(args: Array[String]): Unit ={
    val sparkconf = new SparkConf().setAppName("repartition data")
    val sc = new SparkContext(sparkconf)
    val inpath = args(0)
    val outpath = args(1)
    val numPartition = args(2).toInt

    val log = LogManager.getRootLogger
    log.setLevel(Level.INFO)
    val data = sc.textFile(inpath).repartition(numPartition)
    data.saveAsTextFile(outpath)

  }
}