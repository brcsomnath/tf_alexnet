require 'image'
require 'nn'

local numOfImages = 300;
local batchSize = 100
local trSize = 3*numOfImages

AllImages = {}
AllLabels = {}
classes = {'left','up','right'}
print(classes[1])
local j = 0
local i=0
local count = 0

local n = 'label_trainX.txt'
local file= io.open(n, "r")
if file then
  for line in file:lines() do
      local name = line
      i = i+1
      local f = io.open(name, "r")
      if f==nil then j=j+1
      else
        count = count + 1
        AllImages[count] = name;
        print(AllImages[count])
        AllLabels[count] = 1;
      end
      if (i-j)==numOfImages then break end
  end
end
print(j)

print(classes[2])
local j = 0
local i=0

local n = 'label_trainY.txt'
local file= io.open(n, "r")
if file then
  for line in file:lines() do
      local name = line
      i = i+1
      local f = io.open(name, "r")
      if f==nil then 
        j=j+1
      else
        count = count + 1
        AllImages[count] = name;
        AllLabels[count] = 2;
      end
      if (i-j)==numOfImages then 
        break end
  end
end

print(j)

print(classes[3])
local j = 0
local i=0

local n = 'label_trainZ.txt'
local file= io.open(n, "r")
if file then
  for line in file:lines() do
      local name = line
      i = i+1
      local f = io.open(name, "r")
      if f==nil then j=j+1
      else
        count = count + 1
        AllImages[count] = name;
        AllLabels[count] = 3;
        --print(AllLabels[count])
      end
      if (i-j)==numOfImages then break end
  end
end

print(j)




model = nn.Sequential()
model:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2)) 
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0)
model:add(nn.ReLU())
--model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3,3,2,2))  
model:add(nn.SpatialConvolution(96,256,5,5,1,1,2,2))  
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
--model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3,3,2,2))    
model:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1))  
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0)
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1)) 
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))  
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2))    

model:add(nn.View(256*6*6))
model:add(nn.Linear(256*6*6, 4096))
--    model.modules[#model.modules].weight:normal(0, 0.005)
--    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
--    model.modules[#model.modules].weight:normal(0, 0.005)
--    model.modules[#model.modules].bias:fill(0.1)
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 3))
--    model.modules[#model.modules].weight:normal(0, 0.01)
--    model.modules[#model.modules].bias:fill(0)
model:add(nn.LogSoftMax())

--print(model);

--print(model:forward(trainData.data[1]))
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.002
trainer.learningRateDecay = 1
trainer.shuffleIndices = 0
trainer.maxIteration = 100
torch.save("road_Alexnet_model.t7",model);
--model = torch.load("road_Alexnet_model.t7");

local labelsShuffle1 = torch.randperm(trSize)
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.Tensor(batchSize,3,227,227);
local target=torch.Tensor(batchSize);


local mean = {}
local sum = {0,0,0}
local std = {}
local counter = 0;
local im=torch.Tensor(3, 227, 227);

for i=1,trSize do
  --print(AllImages[i])
  print(i)
  im = image.load(AllImages[i]);
  for c=1,3 do
    sum[c] = sum[c]+im[{c,{},{}}]:sum();
  end
  counter = counter+(227*227);
end

for c=1,3 do
  mean[c] = sum[c]/counter;
end

sum = {0,0,0}
for i=1, trSize do
  im = image.load(AllImages[i]);
  for c=1,3 do
    im[{c,{},{}}]:add(-mean[c]);
    sum[c] = sum[c]+im[{c,{},{}}]:pow(2):sum();
  end
end

for c=1,3 do
  std[c] = torch.sqrt(sum[c]/counter);
end

print("preprocessing done :)");


while true do
  local currentError_ = 0
  for t = 1,(trSize/batchSize) do
    local currentError = 0;
    for t1 = 1,batchSize do
      t2 = (t-1)*batchSize+t1;
      --print(labelsShuffle1[t2])
      im = image.load(AllImages[labelsShuffle1[t2]]);
      im = image.scale(im, 227, 227, 'bilinear');
      input[t1] = im:clone();
      target[t1] = AllLabels[labelsShuffle1[t2]];
      for i=1,3 do
       -- normalize each channel globally:
       input[{{},i,{},{}}]:add(-mean[i])
       input[{{},i,{},{}}]:div(std[i])
      end
      currentError = criterion:forward(model:forward(input[t1]), target[t1])
      --print(currentError)
      currentError_ = currentError_ + currentError
      model:updateGradInput(input[t1], criterion:updateGradInput(model:forward(input[t1]), target[t1]))
      model:accUpdateGradParameters(input[t1], criterion.gradInput, currentLearningRate)
    end
  end

  currentError_ = currentError_ / trSize
  print("#iteration "..iteration..": current error = "..currentError_);
  iteration = iteration + 1
  currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
  if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
    print("# StochasticGradient: you have reached the maximum number of iterations")
    print("# training error = " .. currentError_)
    break
  end
end


--[[
correct = 0
class_perform = {0,0,0}
class_size = {0,0,0}
for i=1,teSize do
    local groundtruth = testData.labels[i]
    --print('ground '..groundtruth)
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(testData.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
end

print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end
--]]