require 'paths'
require 'nn'
require 'cudnn'
require 'nngraph'
nnlib = cudnn
opt={}
ref={}
opt.nModules = 1
opt.nFeats = 256
opt.nStack = 8
ref.nOutChannels = 17
paths.dofile('Residual.lua')
paths.dofile('hg.lua')
model = createModel()
model:cuda()
--cudnn.convert(model,cudnn)
pretrain = torch.load('model_4.t7')
pretrain:clearState()

layer={5,8}
for j,i in ipairs(layer) do
model.modules[i].modules[1].modules[1].modules[3].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[3].weight)
model.modules[i].modules[1].modules[1].modules[6].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[6].weight)
model.modules[i].modules[1].modules[1].modules[9].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[9].weight)
model.modules[i].modules[1].modules[2].modules[1].weight:copy(pretrain.modules[i].modules[1].modules[2].modules[1].weight)
model.modules[i].modules[1].modules[1].modules[3].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[3].bias)
model.modules[i].modules[1].modules[1].modules[6].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[6].bias)
model.modules[i].modules[1].modules[1].modules[9].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[9].bias)
model.modules[i].modules[1].modules[2].modules[1].bias:copy(pretrain.modules[i].modules[1].modules[2].modules[1].bias)
end

res={9,11,12,14,15,17,18,20,21,22,25,28,31,34}
layer={7}
for j =1,8 do
    for r,k in ipairs(res) do
        table.insert(layer,(j-1)*34 + k)
    end
end
for j,i in ipairs(layer) do
model.modules[i].modules[1].modules[1].modules[3].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[3].weight)
model.modules[i].modules[1].modules[1].modules[6].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[6].weight)
model.modules[i].modules[1].modules[1].modules[9].weight:copy(pretrain.modules[i].modules[1].modules[1].modules[9].weight)
model.modules[i].modules[1].modules[1].modules[3].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[3].bias)
model.modules[i].modules[1].modules[1].modules[6].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[6].bias)
model.modules[i].modules[1].modules[1].modules[9].bias:copy(pretrain.modules[i].modules[1].modules[1].modules[9].bias)
end

layer={2}
for j = 1,7 do
    table.insert(layer, (j-1)*34 + 35)
    table.insert(layer, (j-1)*34 + 40)
end
table.insert(layer, (8-1)*34 + 35)
for j,i in ipairs(layer) do
model.modules[i].weight:copy(pretrain.modules[i].weight)
model.modules[i].bias:copy(pretrain.modules[i].bias)
end

torch.save('mpii_pretrain_bce.t7',model)