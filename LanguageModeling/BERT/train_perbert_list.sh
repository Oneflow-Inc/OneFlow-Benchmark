NUM=${2-1}

file_op() 
{
    mkdir -p $1
    mv -f log_f* $1

    # tar  -zcvf $1.tar.gz  $1
    # rm -rf $1
}
#################################################################################
mkdir -p out
mkdir -p pic
rm -rf out/*
rm -rf pic/*


# ###############################################################################
# #                                f32  adam
# ###############################################################################

# for ((i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
# 	sh train_perbert.sh 0 0 64 1 adam
#     cp -rf log/ log_f32_${i}
# done

# file_op out/bert_f32_pretraining_8gpu_64bs_100iter

# # mkdir bert_f32_pretraining_8gpu_64bs_100iter
# # mv -f log_f32_* bert_f32_pretraining_8gpu_64bs_100iter

# # tar  -zcvf bert_f32_pretraining_8gpu_64bs_100iter.tar.gz  \
# # bert_f32_pretraining_8gpu_64bs_100iter
# # rm -rf bert_f32_pretraining_8gpu_64bs_100iter
# ###############################################################################
# #                                f32  lamb 
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
# 	sh train_perbert.sh 0 0 64 1 lamb
#     cp -rf log/ log_f32_${i}
# done

# file_op out/bert_f32_pretraining_8gpu_64bs_100iter_lamb

# # ###############################################################################
# # #                                f16  adam  
# # ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
# 	sh train_perbert.sh  1 0 64 1 adam
#     cp -rf log/ log_f16_${i}
# done

# file_op out/bert_f16_pretraining_8gpu_64bs_100iter

# ###############################################################################
# #                                f16  lamb  
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
#     sh train_perbert.sh  1 0 64 1 lamb
#     cp -rf log/ log_f16_${i}
# done

# file_op out/bert_f16_pretraining_8gpu_64bs_100iter_lamb

# ###############################################################################
# #                             f32 accumulation adam
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
#     sh train_perbert.sh 0 0 32 2 adam
#     cp -rf log/ log_f32_${i}
# done

# file_op out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation

# ###############################################################################
# #                             f32 accumulation lamb
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
#     sh train_perbert.sh 0 0 32 2 lamb
#     cp -rf log/ log_f32_${i}
# done

# file_op out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_lamb

# ###############################################################################
# #                             f16 accumulation adam
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
#     sh train_perbert.sh 1 0 32 2 adam
#     cp -rf log/ log_f16_${i}
# done

# file_op out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation

# ###############################################################################
# #                             f16 accumulation lamb
# ###############################################################################

# for (( i = 1; i <= ${NUM}; i++ ))
# do
#     echo $i
#     sh train_perbert.sh 1 0 32 2 lamb
#     cp -rf log/ log_f16_${i}
# done

# file_op out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_lamb


###############################################################################
#                                f32  adam debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
	sh train_perbert.sh 0 1 64 1 adam
    cp -rf log/ log_f32_${i}
done

file_op out/bert_f32_pretraining_8gpu_64bs_100iter_debug

python tools/result_analysis.py  --f32=1 \
    --cmp1_file=old/bert_f32_pretraining_8gpu_64bs_100iter_debug/log_f32_1/out.json \
    --cmp2_file=out/bert_f32_pretraining_8gpu_64bs_100iter_debug/log_f32_1/out.json \
    --out=pic/bert_f32_pretraining_8gpu_64bs_100iter_debug.png
###############################################################################
#                                f32  lamb debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
	sh train_perbert.sh 0 1 64 1 lamb
    cp -rf log/ log_f32_${i}
done

file_op  out/bert_f32_pretraining_8gpu_64bs_100iter_lamb_debug

python tools/result_analysis.py  --f32=1 \
    --cmp1_file=old/bert_f32_pretraining_8gpu_64bs_100iter_lamb_debug/log_f32_1/out.json \
    --cmp2_file=out/bert_f32_pretraining_8gpu_64bs_100iter_lamb_debug/log_f32_1/out.json \
    --out=pic/bert_f32_pretraining_8gpu_64bs_100iter_lamb_debug.png
###############################################################################
#                                f16  adam debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
	sh train_perbert.sh 1 1 64 1 adam
    cp -rf log/ log_f16_${i}
done
file_op  out/bert_f16_pretraining_8gpu_64bs_100iter_debug

python tools/result_analysis.py  --f32=0 \
    --cmp1_file=old/bert_f16_pretraining_8gpu_64bs_100iter_debug/log_f16_1/out.json \
    --cmp2_file=out/bert_f16_pretraining_8gpu_64bs_100iter_debug/log_f16_1/out.json \
    --out=pic/bert_f16_pretraining_8gpu_64bs_100iter_debug.png
###############################################################################
#                                f16  lamb debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
    sh train_perbert.sh 1 1 64 1 lamb
    cp -rf log/ log_f16_${i}
done

file_op  out/bert_f16_pretraining_8gpu_64bs_100iter_lamb_debug

python tools/result_analysis.py  --f32=0 \
    --cmp1_file=old/bert_f16_pretraining_8gpu_64bs_100iter_lamb_debug/log_f16_1/out.json \
    --cmp2_file=out/bert_f16_pretraining_8gpu_64bs_100iter_lamb_debug/log_f16_1/out.json \
    --out=pic/bert_f16_pretraining_8gpu_64bs_100iter_lamb_debug.png
###############################################################################
#                             f32 accumulation adam debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
    sh train_perbert.sh 0 1 32 2 adam
    cp -rf log/ log_f32_${i}
    
done

file_op out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_debug

python tools/result_analysis.py  --f32=1 \
    --cmp1_file=old/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_debug/log_f32_1/out.json \
    --cmp2_file=out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_debug/log_f32_1/out.json \
    --out=pic/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_debug.png

###############################################################################
#                             f32 accumulation lamb debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
    sh train_perbert.sh 0 1 32 2 lamb
    cp -rf log/ log_f32_${i}
done

file_op out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug

python tools/result_analysis.py  --f32=1 \
    --cmp1_file=old/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug/log_f32_1/out.json \
    --cmp2_file=out/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug/log_f32_1/out.json \
    --out=pic/bert_f32_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug.png

###############################################################################
#                             f16 accumulation adam debug
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
    sh train_perbert.sh 1 1 32 2 adam
    cp -rf log/ log_f16_${i}
done

file_op out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_debug

python tools/result_analysis.py  --f32=0 \
    --cmp1_file=old/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_debug/log_f16_1/out.json \
    --cmp2_file=out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_debug/log_f16_1/out.json \
    --out=pic/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_debug.png
###############################################################################
#                             f16 accumulation lamb
###############################################################################

for (( i = 1; i <= ${NUM}; i++ ))
do
    echo $i
    sh train_perbert.sh 1 1 32 2 lamb
    cp -rf log/ log_f16_${i}
done

file_op out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug

python tools/result_analysis.py  --f32=0 \
    --cmp1_file=old/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug/log_f16_1/out.json \
    --cmp2_file=out/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug/log_f16_1/out.json \
    --out=pic/bert_f16_pretraining_8gpu_64bs_100iter_accumulation_lamb_debug.png
# ##############################################################################
#                             tar
# ##############################################################################
    
tar  -zcvf out.tar.gz  out

python  tools/stitching_pic.py --dir=pic --out_file=./pic/all.png
# rm -rf out
###############################################################################
#                              upload
###############################################################################





