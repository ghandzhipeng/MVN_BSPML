#! /bin/bash
reg_para=0
cores_per_executor=1
partitions_per_core=1
whether_debug=true
num_worker=8

func(){
    bash /mnt/local/zhipeng/detail-analysis/clean_mylog.sh
    inf=$1
    num_iter=$2
    budget=$3
    steps=$4
    num_features=$5
    data_name=$6

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
            --class svmWithLBFGS /mnt/local/zhipeng/ghandMLlib.jar\
            ${inf} ${num_iter} ${budget} ${steps} ${num_worker} ${cores_per_executor} ${partitions_per_core} ${reg_para} ${num_features}

    bash /mnt/local/zhipeng/detail-analysis/gather_mylog.sh self-logs/linearSVC-${data_name}-${num_worker}-${reg_para}
#sleep 10
}
# usage:
#func $input_path ${num_iteration} ${budget} ${step_size} ${data_name}

## default paras
input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/avazu-full.partition
num_iteration=20
step_size=1
budget=6
#cores_per_executor=6
#partition_per_core=1
num_features=999990
#func ${input_path} ${num_iteration} ${budget} ${step_size} ${num_features} avazu

## default paras
input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/url_combined.partition
num_iteration=5
step_size=1
budget=10
#cores_per_executor=6
#partition_per_core=1
num_features=3231961
func ${input_path} ${num_iteration} ${budget} ${step_size} ${num_features} url

## default paras
input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/kddb.partition
num_iteration=5
step_size=1
budget=10
#cores_per_executor=1
#partition_per_core=1
num_features=29890095
#func ${input_path} ${num_iteration} ${budget} ${step_size} ${num_features} kddb

## default paras
input_path=hdfs://bach03:9000/user/zhanzhip/MLBench/kdd12.partition
num_iteration=20
step_size=1
budget=10
#cores_per_executor=1
#partition_per_core=1
num_features=54686452

#func ${input_path} ${num_iteration} ${budget} ${step_size} ${num_features} kdd12
