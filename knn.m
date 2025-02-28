function [allImages, labels] = dataPreprocessing(folderPaths)
    % Input: folderPaths - cell array of folder paths (e.g., {'path1', 'path2', 'path3'})
    % Output: allImages - cell array of processed images
            % labels - vector of labels corresponding to the folder index, size [numImages x 1]

    allImages = []; % Initialize an empty cell array to store processed images
    labels = [];

    % Iterate over each folder path
    for i = 1:length(folderPaths)
        folderPath = folderPaths{i}; % Get the current folder path
        [~, folderName, ~] = fileparts(folderPath);
        % Get a list of all image files in the folder
        imageFiles = [dir(fullfile(folderPath, '*.jpg')); 
                     dir(fullfile(folderPath, '*.png')); 
                     dir(fullfile(folderPath, '*.bmp')); 
                     dir(fullfile(folderPath, '*.jpeg'))];
        disp(folderPath)
        disp(length(imageFiles))
        % Process each image in the folder
        for j = 1:length(imageFiles)
            % Read the image
            imgPath = fullfile(folderPath, imageFiles(j).name);
            img = imread(imgPath);
            
            % Resize the image to 100x100
            resizedImg = imresize(img, [100, 100]);
            
            % Convert the image to grayscale
            grayImg = im2gray(resizedImg);
            
            % Flatten the grayscale image into a row vector
            flattenedImg = reshape(grayImg, 1,[]); % 1x10000 vector
           
            %  Append the flattened image to allImages
            allImages = [allImages; double(flattenedImg)]; % Add to matrix
            
            % Append the corresponding label (folder index) to labels
            labels = [labels; categorical({folderName})];
        end
    end
     % Display completion message
    fprintf('Data preprocessing completed. Total images processed: %d\n', size(allImages, 1));
end


% Preprocessing
folderPaths = {'DataSet/Animal', 'DataSet/Object', 'DataSet/Person'};
[processedData, labels] = dataPreprocessing(folderPaths);

% Mean normalization
meanValues = mean(processedData, 1);
normalizedData = processedData - meanValues;

% Covariance
covMat = cov(normalizedData);

%% Eigenvalues and eigenvectors of the covariance matrix
[eigenVectors, eigenValuesMatrix] = eig(covMat);
eigenValues = diag(eigenValuesMatrix); % Extract eigenvalues into a vector
[eigenValuesSorted, sortOrder] = sort(eigenValues, 'descend'); % Sort eigenvalues in descending order
eigenVectorsSorted = eigenVectors(:, sortOrder); % Sort eigenvectors accordingly
% disp(eigenValuesSorted)
% size(eigenVectorsSorted);

%% Finding top k eigen vectors that cover 95% of variance
totalVariance = sum(eigenValuesSorted);
cumulativeVariance = cumsum(eigenValuesSorted) / totalVariance;
%%
k = find(cumulativeVariance >= 0.75, 1);
% disp(k)
% disp(cumulativeVariance(1:50,:))
%%
topEigenVectors = eigenVectorsSorted(:, 1:k);  % Getting top k eigenvectors

%% Project the data onto the top k eigenvectors - Dimension Reduction
projectedData = normalizedData * topEigenVectors;

% Number of folds for cross-validation
kFolds = 5;
%numSamples = size(projectedData, 1);
indices = crossvalind('Kfold', labels, kFolds); % Create fold indices
accuracies = zeros(kFolds, 1); % Store accuracies for each fold
modelToUse =[];
earlierAccuracy = 0;
% k-fold cross-validation
for fold = 1:kFolds
    fprintf('Processing fold %d/%d...\n', fold, kFolds);
    
    % Separate training and testing data for the current fold
    testIdx = (indices == fold); % Indices for the test set
    trainIdx = ~testIdx; % Indices for the training set
    
    trainData = projectedData(trainIdx, :);
    testData = projectedData(testIdx, :);
    trainLabels = labels(trainIdx, :);
    testLabels = labels(testIdx, :);
    
    % Train KNN model
    knnModel = fitcknn(trainData, trainLabels, 'NumNeighbors', 3); % k=3
    
    % Predict on test data
    predictedLabels = predict(knnModel, testData);
    
    % Calculate accuracy for the current fold
    correctPredictions = sum(predictedLabels == testLabels);
    foldAccuracy = (correctPredictions / length(testLabels)) * 100;
    if (foldAccuracy > earlierAccuracy)
        earlierAccuracy = foldAccuracy;
        modelToUse = knnModel;
    end

    accuracies(fold) = foldAccuracy; % Store accuracy
    
    fprintf('Accuracy for fold %d: %.2f%%\n', fold, foldAccuracy);
end

% Compute mean accuracy across all folds
meanAccuracy = mean(accuracies);
fprintf('Mean Accuracy across all folds: %.2f%%\n', meanAccuracy);

%{
%uncomment this part to save the model and build a Raspberry pi object
saveLearnerForCoder(modelToUse,'TestProject');

%% step 7.0

  %step 7.1
r = raspi('<IP of Raspberry PI>', '<username>', '<password>');
%}

%% remove

imgT = imread('LiveTesting/demo1.jpg');
            
% Resize the image to 100x100
resizedImgT = imresize(imgT, [100, 100]);

% Convert the image to grayscale
grayImgT = im2gray(resizedImgT);

% Flatten the grayscale image into a row vector
flattenedImgT = reshape(grayImgT, 1,[]); % 1x10000 vector
flattenedImgT  = double(flattenedImgT);

% Normalize the reduced data
normalizedDataT = flattenedImgT - meanValues;

% Project the data onto the top k eigenvectors - Dimension Reduction
projectedDataT = normalizedDataT * topEigenVectors;
%Mdl = loadLearnerForCoder('TestProject.mat');
label = predict(modelToUse,projectedDataT)
%%
cprintf('*[1,1,1]', 'classifyImagesProject\n');  % Displays bold red text
