require 'image'
require 'nn'

local numOfImages = 200;
local imageAll = torch.Tensor(3*numOfImages,3,227,227)
local labelAll = torch.Tensor(3*numOfImages)
local batchSize = 100
classes = {'left','up','right'}
print(classes[1])
local j = 0
local i=0
while 1 do
   i=i+1
   local name = 'data/train/Left_screenshot'..i..'_1.png'
   local f= io.open(name, "r")
   if f==nil then j=j+1
   else
	  im = image.load(name);
    im = image.scale(im, 227, 227, 'bilinear');
    imageAll[i-j] = im;
	  labelAll[i-j] = 1;
   end
   if (i-j)==numOfImages then break end
end
print(j)

local j=0
local i=0
while 1 do
   i=i+1
   local name = 'data/train/Up_screenshot'..i..'_1.png'
   local f= io.open(name, "r")
   if f==nil then j=j+1
   else
     im = image.load(name);
     im = image.scale(im, 227, 227, 'bilinear');
     imageAll[i + numOfImages-j] = im;
     labelAll[i + numOfImages-j] = 2;
   end
   if (i-j)==numOfImages then break end
end

local j =0
local i=0
while 1 do
   i=i+1
   local name = 'data/train/Right_screenshot'..i..'_1.png'
   local f= io.open(name, "r")
   if f==nil then j=j+1
   else
      im = image.load(name);
      im = image.scale(im, 227, 227, 'bilinear');
      imageAll[2*numOfImages+i-j] = im;
      labelAll[2*numOfImages+i-j] = 3;
   end
   if (i-j)==numOfImages then break end
end

for i=1,3*numOfImages do
   if labelAll[i]<1 then print(i) end
   if labelAll[i]>3 then print('>3'..i)
   end
end

local labelsShuffle = torch.randperm((#labelAll)[1])

local portionTrain = 0.9 	--90% data is trained
local trSize = torch.floor(labelsShuffle:size(1)*portionTrain)
local teSize = labelsShuffle:size(1) - trSize

trainData = {
	data = torch.Tensor(trSize,3,227,227),
	labels = torch.Tensor(trSize),
	size = function() return trSize end
}

testData = {
	data = torch.Tensor(teSize,3,227,227),
	labels = torch.Tensor(teSize),
	size = function() return teSize end
}

for i=1,trSize do
	trainData.data[i] = imageAll[labelsShuffle[i]]:clone()
	trainData.labels[i] = labelAll[labelsShuffle[i]]
end

for i=trSize+1,trSize+teSize do
	testData.data[i-trSize] = imageAll[labelsShuffle[i]]:clone()
	testData.labels[i-trSize] = labelAll[labelsShuffle[i]]
end


setmetatable(trainData,
    {__index = function(t, i)
                    return {t.data[i], t.labels[i]}
                end}
);
trainData.data = trainData.data:double()

function trainData:size()
    return self.data:size(1)
end

local mean = {}
local std = {}

for i=1,3 do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

for i=1,3 do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

for i=1,3 do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..i..'-channel, mean: ' .. trainMean)
   print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..i..'-channel, mean: ' .. testMean)
   print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end

-- visualtisation part may be deleted. check before final run
--if opt.visualize then
--   local first256Samples_y = trainData.data[{ {1,256},1 }]
--   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
--   local first256Samples_y = testData.data[{ {1,256},1 }]
--   image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
--end


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
trainer.maxIteration = 20
torch.save("road_Alexnet_model.t7",model);
--model = torch.load("road_Alexnet_model.t7");

local labelsShuffle1 = torch.randperm(trSize)
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.Tensor(batchSize,3,227,227);
local target=torch.Tensor(batchSize);
while true do
  local currentError_ = 0
  for t = 1,(trSize/batchSize) do
    local currentError = 0;
    for t1 = 1,batchSize do
      t2 = (t-1)*batchSize+t1;
      print("nigga")
      print(labelsShuffle1[t2])
      input[t1] = trainData.data[labelsShuffle1[t2]];
      target[t1] = trainData.labels[labelsShuffle1[t2]]
      for i=1,3 do
       -- normalize each channel globally:
       input[{{},i,{},{}}]:add(-mean[i])
       input[{{},i,{},{}}]:div(std[i])
      end
      currentError = currentError + criterion:forward(model:forward(input[t1]), target[t1])
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