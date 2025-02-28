function label = classifyImagesProject(path,topEigenVectors,meanValues) %#codegen
% Data pre processing
imgT = imread(path);
disp("Read image")            
% Resize the image to 100x100
resizedImgT = imresize(imgT, [100, 100]);
disp("Resized image") 
% Convert the image to grayscale
grayImgT = im2gray(resizedImgT);
disp("grayed image") 
% Flatten the grayscale image into a row vector
flattenedImgT = reshape(grayImgT, 1,[]); % 1x10000 vector
flattenedImgT  = double(flattenedImgT);
disp("flattened image") 
% Normalize the reduced data
normalizedDataT = flattenedImgT - meanValues;
disp("normal image") 
% Project the data onto the top k eigenvectors - Dimension Reduction
projectedDataT = normalizedDataT * topEigenVectors;
disp("project image") 
Mdl = loadLearnerForCoder('TestProject.mat');
disp("trying to predict image") 
label = predict(Mdl,projectedDataT);
end