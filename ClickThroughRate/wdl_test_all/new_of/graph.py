import oneflow as flow


class WideAndDeepGraph(flow.nn.Graph):
    def __init__(self, args,wdl_module, dataloader, bce_loss):
        super(WideAndDeepGraph, self).__init__()
        self.args=args
        self.module = wdl_module
        self.dataloader = dataloader
        self.bce_loss = bce_loss

    def build(self):
        with flow.no_grad():
            return self.graph()

    def graph(self):
        (
            labels,
            dense_fields,
            wide_sparse_fields,
            deep_sparse_fields,
        ) = self.dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")

        predicts = self.module(dense_fields, wide_sparse_fields, deep_sparse_fields)
        loss = self.bce_loss(predicts, labels)
        loss=loss/self.args.batch_size
        return predicts, labels, loss


class WideAndDeepTrainGraph(WideAndDeepGraph):
    def __init__(self, args,wdl_module, dataloader, bce_loss, optimizer):
        super(WideAndDeepTrainGraph, self).__init__(args,wdl_module, dataloader, bce_loss)
        self.add_optimizer(optimizer)

    def build(self):
        predicts, labels, loss = self.graph()
        loss.backward()
        return predicts, labels, loss
