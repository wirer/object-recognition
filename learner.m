% Step1: reading Data from the file
file_data = load('Features.txt');
Data = file_data(:,1:end-1)';
Labels = file_data(:, end)';
Labels = Labels*2 - 1;
MaxIter = 200; % boosting iterations
% Step2: splitting data to training and control set
TrainData = Data(:,1:2:end);
TrainLabels = Labels(1:2:end);
ControlData = Data(:,2:2:end);
ControlLabels = Labels(2:2:end);
% Step3: constructing weak learner
weak_learner = tree_node_w(3); % pass the number of tree splits to the constructor

% Step4: training real adaboost
[RLearners RWeights] = RealAdaBoost(weak_learner, TrainData, TrainLabels, 1, RWeights, RLearners);

% Step5: evaluating on control set
ResultR = sign(Classify(RLearners, RWeights, ControlData));

% Step6: calculating error
ErrorR = sum(ControlLabels ~= ResultR)
