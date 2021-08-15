CMP_NEW=${1:-python_whl}
CMP_OLD=${2:-325160bcfb786b166b063e669aea345fadee2da7}

BERT_OSSDIR=oss://oneflow-public/BERT/
####################################################################
#                               
####################################################################

echo ${CMP_NEW}
echo ${CMP_OLD}

pip install $CMP_NEW
####################################################################
#                           
####################################################################


cd /root/OneFlow-Benchmark/LanguageModeling/BERT

ossutil64 cp ${BERT_OSSDIR}$CMP_OLD/out.tar.gz .

tar xvf out.tar.gz
rm -rf old
mv out old
rm out.tar.gz

bash train_perbert_list.sh

####################################################################
#                       
####################################################################
ossutil64 rm -rf  ${BERT_OSSDIR}$CMP_NEW/*
ossutil64 mkdir ${BERT_OSSDIR}$CMP_NEW/


ossutil64 cp out.tar.gz ${BERT_OSSDIR}$CMP_NEW/
ossutil64 cp -rf pic ${BERT_OSSDIR}$CMP_NEW/

echo "success"
