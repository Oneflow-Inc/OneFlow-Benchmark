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

for host in "${hosts[@]}" 
do
  ssh $USER@$host "docker kill oneflow-auto-test"

done
