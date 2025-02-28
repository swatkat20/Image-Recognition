%% step 1
cfg = coder.config('lib', 'ecoder', true);
cfg.VerificationMode = 'PIL';

%% step 2
hw = coder.hardware('Raspberry Pi');
cfg.Hardware = hw;

%% step 3
buildDir = '/home/pr/projectTest';
cfg.Hardware.BuildDir = buildDir;

%% step 4
filePath = coder.typeof(char(0), [1,255], [false true]);
topEigenVec = coder.typeof(zeros(10000, 515),[10000,515],[false false]);
meanVal = coder.typeof(zeros(1, 10000),[1,10000],[false false]);
codegen classifyImagesProject -config cfg -args {filePath,topEigenVectors,meanVal};