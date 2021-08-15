CMP_NEW=${1:-python_whl}
CMP_OLD=${2:-325160bcfb786b166b063e669aea345fadee2da7}

BERT_OSSDIR=https://oneflow-public.oss-cn-beijing.aliyuncs.com/BERT/
LOGFILE=1n_out.tar.gz
####################################################################
#                               
####################################################################

pip install $CMP_NEW

echo ${CMP_NEW}
echo ${CMP_OLD}
echo $(pwd)
####################################################################
#                           
####################################################################
cd OneFlow-Benchmark/LanguageModeling/BERT
echo $(pwd)

wget ${BERT_OSSDIR}${CMP_OLD}/${LOGFILE}

echo $(pwd)
tar xvf ${LOGFILE}
rm -rf old
mv out old

echo $(pwd)
bash train_prebert_list.sh

####################################################################
#                       
####################################################################
# ossutil64 rm -rf  ${BERT_OSSDIR}$CMP_NEW/*
# ossutil64 mkdir ${BERT_OSSDIR}$CMP_NEW/


# ossutil64 cp out.tar.gz ${BERT_OSSDIR}$CMP_NEW/
# ossutil64 cp -rf pic ${BERT_OSSDIR}$CMP_NEW/

echo "success"
