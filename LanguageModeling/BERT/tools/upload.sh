CMP_NEW=${1:-python_whl}
BERT_OSSDIR=oss://oneflow-public/BERT/
LOGFILE=1n_out.tar.gz

tar  -zcvf ${LOGFILE}  OneFlow-Benchmark/LanguageModeling/BERT/out

ossutil64 rm -rf  ${BERT_OSSDIR}$CMP_NEW/*
ossutil64 mkdir ${BERT_OSSDIR}$CMP_NEW/


ossutil64 cp ${LOGFILE} ${BERT_OSSDIR}$CMP_NEW/
ossutil64 cp -rf OneFlow-Benchmark/LanguageModeling/BERT/pic ${BERT_OSSDIR}$CMP_NEW/

echo "success"
