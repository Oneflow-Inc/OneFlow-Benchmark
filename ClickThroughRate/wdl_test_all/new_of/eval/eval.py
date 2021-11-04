import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
import oneflow._C as F
from tqdm import tqdm
from config import get_args
from models.dataloader_utils import OFRecordDataLoader
from oneflow.framework import distribute
from models.wide_and_deep import WideAndDeep
from oneflow.nn.parallel import DistributedDataParallel as ddp
from graph import WideAndDeepGraph,WideAndDeepTrainGraph
import warnings
import pandas as pd
from datetime import datetime
from pathlib import Path


class Trainer(object):
    def __init__(self,args):
        self.args = args
        self.batch_size=args.batch_size
        self. test_name=args.test_name
        self.execution_mode = args.execution_mode
        self.ddp = args.ddp
        if self.ddp == 1 and self.execution_mode == "graph":
            warnings.warn(
                """when ddp is True, the execution_mode can only be eager, but it is graph""",
                UserWarning,
            )
            self.execution_mode = "eager"
        self.is_consistent = (
            flow.env.get_world_size() > 1 and not args.ddp
        ) or args.execution_mode == "graph"
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        (
            self.train_dataloader,
            self.val_dataloader,
            self.wdl_module,
            self.loss,
            self.opt,
        ) = self.prepare_modules()
        if self.execution_mode == "graph":
            self.train_graph = WideAndDeepTrainGraph(
                self.args,self.wdl_module, self.train_dataloader, self.loss, self.opt
            )
        self.record=[]

    def get_memory_usage(self):
        currentPath=os.path.dirname(os.path.abspath(__file__))
        dir = Path(os.path.join(currentPath,'csv/gpu_info'))
        if not dir.is_dir():
            os.makedirs(dir) 
        nvidia_smi_report_file_path=os.path.join('csv/gpu_info','gpu_memory_usage_%s.csv'%self.rank)
        nvidia_smi_report_file_path=os.path.join(currentPath,nvidia_smi_report_file_path)
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
        if nvidia_smi_report_file_path is not None:
            cmd += f" -f {nvidia_smi_report_file_path}"
        os.system(cmd)
        df=pd.read_csv(nvidia_smi_report_file_path)
        memory=df.iat[self.rank,1].split()[0]
        return memory

    def record_to_csv(self):
        dir_path=os.path.join('/home/shiyunxiao/of_benchmark/OneFlow-Benchmark/ClickThroughRate/wdl_test_all/results/new/%s'%(self. test_name))
        isExists=os.path.exists(dir_path)
        if not isExists:
             os.makedirs(dir_path) 
        filePath=os.path.join(dir_path,'record_%s_%s.csv'%(self.args.batch_size,self.rank))
        df_record=pd.DataFrame.from_dict(self.record, orient='columns')
        df_record.to_csv(filePath,index=False)

    def to_record(self,iter=0,loss=0,latency=0):
        data={}
        data['node']=1
        data['device']=self.rank
        data['batch_size']=self.args.batch_size
        data['deep_vocab_size']=self.args.deep_vocab_size
        data['deep_embedding_vec_size']=self.args.deep_embedding_vec_size
        data['hidden_units_num']=self.args.hidden_units_num
        data['iter']=iter
        data['latency/ms']=latency
        data['memory_usage/MB']=self.get_memory_usage()
        data['loss']=loss      
        self.record.append(data)

    def prepare_modules(self):
        args = self.args
        is_consistent = self.is_consistent
        self.wdl_module = WideAndDeep(args)
        if is_consistent == True:
            world_size = self.world_size
            placement = flow.placement("cuda", {0: range(world_size)})
            self.wdl_module = self.wdl_module.to_consistent(
                placement=placement, sbp=flow.sbp.broadcast
            )
        else:
            self.wdl_module=self.wdl_module.to("cuda")
        if args.model_load_dir != "":
            self.load_state_dict()
        if self.ddp:
            self.wdl_module = ddp(self.wdl_module)
        if args.save_initial_model and args.model_save_dir != "":
            self.save(os.path.join(args.model_save_dir, "initial_checkpoint"))

        train_dataloader = OFRecordDataLoader(args)
        val_dataloader = OFRecordDataLoader(args, mode="val")

        #新版里面用的 flow.nn.BCELoss 这个 module，背后是新写的 binary_cross_entropy op & kernel，这个 op 自带 reduction 属性，kernel 实现的时候没考虑 consistent 的并行情况。
        '''
        weight (oneflow.Tensor, optional) – The manual rescaling weight to the loss. Default to None, whose corresponding weight value is 1.
        reduction (str, optional) – The reduce type, it can be one of “none”, “mean”, “sum”. Defaults to “mean”.
        '''
        bce_loss = flow.nn.BCELoss(reduction="sum")
        bce_loss.to("cuda")

        opt = flow.optim.SGD(
            self.wdl_module.parameters(), lr=args.learning_rate, momentum=0.9
        )

        return train_dataloader, val_dataloader, self.wdl_module, bce_loss, opt

    def load_state_dict(self):
        print(f"Loading model from {self.args.model_load_dir}")
        if self.is_consistent:
            state_dict = flow.load(self.args.model_load_dir, consistent_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.args.model_load_dir)
        else:
            return
        self.wdl_module.load_state_dict(state_dict)

    def save(self, save_path):
        if save_path is None:
            return
        print(f"Saving model to {save_path}")
        state_dict = self.wdl_module.state_dict()
        if self.is_consistent:
            flow.save(state_dict, save_path, consistent_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return

    def __call__(self):
        self.train()

    def train(self):
        def handle(dict):
            for key, value in dict.items():
                if self.is_consistent == True:
                    dict[key] = (
                        value.to_consistent(
                            placement=flow.placement(
                                "cuda", {0: range(self.world_size)}
                            ),
                            sbp=flow.sbp.broadcast,
                        )
                        .to_local()
                        .numpy()
                    )
                else:
                    dict[key] = value.numpy()
            return dict

        losses = []
        args = self.args
        latency=0
        time_begin=time.time()
        tmp_latency_list=[]
        for i in range(args.max_iter):
            loss = self.train_one_step()
            time_end=time.time()
            tmp_latency=(time_end-time_begin)*1000/args.print_interval
            tmp_latency_list.append(tmp_latency)
            losses.append(handle({"loss": loss})["loss"])
            time_begin=time.time()
            if (i + 1) % args.print_interval == 0:
                time_end=time.time()
                tmp_latency=(time_end-time_begin)*1000/args.print_interval
                tmp_latency_list.append(tmp_latency)
                l = sum(losses) / len(losses)
                latency=np.sum(tmp_latency_list)
                self.to_record(i+1,l,round(latency,3))
                tmp_latency_list=[]
                losses = []
                latency=0
                time_begin=time.time()   
            
        self.record_to_csv()
    
    def train_eager(self):
        def forward():
            (
                labels,
                dense_fields,
                wide_sparse_fields,
                deep_sparse_fields,
            ) = self.train_dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields
            )

            train_loss = self.loss(predicts,labels)
            train_loss=train_loss/self.batch_size

            return predicts,labels,train_loss
        predicts,labels,loss = forward()

        if loss.is_consistent:
            # NOTE(zwx): scale init grad with world_size
            # because consistent_tensor.mean() include dividor numel * world_size
            loss.backward()
            for param_group in self.opt.param_groups:
                for param in param_group.parameters:
                    param.grad *= self.world_size             
        else:
            loss.backward()
            # loss/=self.world_size
        self.opt.step()
        self.opt.zero_grad()

        return predicts,labels,loss

    def train_one_step(self):
        self.wdl_module.train()
        if self.execution_mode == "graph":
            predicts, labels, train_loss = self.train_graph()
        else:
            predicts, labels, train_loss = self.train_eager()
        return train_loss
        


if __name__ == "__main__":
    flow.distributed.launch.main()
    trainer = Trainer()
    trainer()
