import oneflow as flow

placement = flow.placement("cuda",{0:[0,1]})
sbp = flow.sbp.split(0)
x = flow.randn(2,5,placement=placement,sbp=sbp)
print('x.shape',x.shape)
print('x.sbp',x.sbp)
# print('x.to_local().numpy()',x.to_local().numpy())
#print('x.numpy()',x.numpy())
print('x',x)
