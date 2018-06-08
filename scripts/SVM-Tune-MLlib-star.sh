#! /bin/bash
cores_per_executor=1
partition_per_core=1
num_worker=8
reg_para=0
triers=8
whether_debug=true

func(){
    bash /mnt/local/zhipeng/detail-analysis/clean_mylog.sh
    inf=$1
    num_features=$2
    data_name=$3

    /mnt/local/zhipeng/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master yarn\
            --conf spark.eventLog.enabled=true\
            --deploy-mode cluster\
            --num-executors ${num_worker} \
            --executor-cores ${cores_per_executor} \
            --conf spark.driver.memory=20g\
            --conf spark.rpc.message.maxSize=500 \
            --conf spark.reducer.maxSizeInFlight=144m\
            --conf spark.shuffle.file.buffer=32k\
            --conf spark.driver.cores=8\
            --conf spark.driver.maxResultSize=15g\
            --conf spark.memory.fraction=0.7\
            --conf spark.locality.wait=1s\
            --conf spark.executor.heartbeatInterval=30s\
            --conf spark.ml.useFeatureScaling=true \
            --conf spark.ml.numClasses=2 \
            --conf spark.ml.debug=${whether_debug} \
            --conf spark.ml.straggler=false \
            --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC"\
            --executor-memory 20g\
            --class svmTuneMLlibStar /mnt/local/zhipeng/ghandMLlib.jar\
            ${inf} ${num_worker} ${reg_para} ${triers} ${num_features}

    bash /mnt/local/zhipeng/detail-analysis/gather_mylog.sh self-logs/MLlibStar-tune-${data_name}-${reg_para}
}
# usage: func infile

## default paras
input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/avazu-full.partition
num_features=999990
#func ${input_path} avazu

input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/url_combined.partition
num_features=3231961
#func ${input_path} url

input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/kddb.partition
num_features=29890095
func ${input_path} kddb

input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/kdd12.partition
num_features=54686452
#func ${input_path} kdd12

#input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/real-sim.spark
#input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/ijcnn
#input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/webspam_wc_normalized_trigram.svm.partition
