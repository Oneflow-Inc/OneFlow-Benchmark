NUM_NODES=${1-4}


#######################################################################################
#                     0 prepare the host list ips for training
########################################################################################
declare -a host_list=("10.11.0.2" "10.11.0.3" "10.11.0.4" "10.11.0.5")

if [ $NUM_NODES -gt ${#host_list[@]} ]
then 
    echo num_nodes should be less than or equal to length of host_list.
    exit
fi

hosts=("${host_list[@]:0:${NUM_NODES}}")
echo "Working on hosts:${hosts[@]}"

ips=${hosts[0]}
for host in "${hosts[@]:1}" 
do
   ips+=",${host}"
done

#######################################################################################
#                     1 prepare docker image
########################################################################################
WORK_PATH=`pwd`

wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/bert/docker_image/oneflow_autobert.tar

for host in "${hosts[@]}" 
do
  ssh $USER@$host "mkdir -p ~/oneflow_docker_temp; rm -rf ~/oneflow_docker_temp/*"
  scp -r oneflow_autobert.tar  $USER@$host:~/oneflow_docker_temp
  ssh $USER@$host " docker load --input ~/oneflow_docker_temp/oneflow_autobert.tar; "

  echo "tesst--->"
  ssh $USER@$host "  \
        docker run  --runtime=nvidia --rm -i -d --privileged --shm-size=16g \
            --ulimit memlock=-1 --net=host  \
            --name oneflow-auto-test   \
            --cap-add=IPC_LOCK --device=/dev/infiniband  \
            -v /data/bert/:/data/bert/ \
            -v /datasets/bert/:/datasets/bert/  \
            -v /datasets/ImageNet/OneFlow/:/datasets/ImageNet/OneFlow/ \
            -v /data/imagenet/ofrecord:/data/imagenet/ofrecord \
            -v ${WORK_PATH}:/workspace/oneflow-test \
            -w /workspace/oneflow-test  \
            oneflow:cu11.2-ubuntu18.04 bash   -c \"/usr/sbin/sshd -p 57520 && bash\" "
done
