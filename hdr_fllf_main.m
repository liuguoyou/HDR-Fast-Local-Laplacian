%% hdr recovery
dirName = ('indoor\');
[filenames, exposures, numExposures] = readDir(dirName);
tmp = imread(filenames{1});
numPixels = size(tmp,1) * size(tmp,2);

% define lamda smoothing factor
l = 10;

% precompute the weighting function value for each pixel
weights = [0:1:127, 127:-1:0];

% creating exposures matrix B
B = zeros(1, numExposures);
for i = 1:numExposures
    B(1,i) = log(exposures(i)); %log delta t
end

% flag for solving camera response function
solve_crf = 0;

if solve_crf == 1
    % load and sample the images
    [zRed, zGreen, zBlue, sampleIndices] = makeImageMatrix(filenames, numPixels);
    % solve the system for each color channel
    [gRed,lERed]=gsolve(zRed, B, l, weights);
    [gGreen,lEGreen]=gsolve(zGreen, B, l, weights);
    [gBlue,lEBlue]=gsolve(zBlue, B, l, weights);
    save('gRed.mat', 'gRed');
    save('gGreen.mat', 'gGreen');
    save('gBlue.mat', 'gBlue');
    plot(gRed, 'r');
    hold on
    plot(gBlue, 'b');
    plot(gGreen, 'g');
else
    load('gRed.mat', 'gRed');
    load('gGreen.mat', 'gGreen');
    load('gBlue.mat', 'gBlue');
end

% compute the hdr radiance map
I = hdr(filenames, gGreen, gGreen, gGreen, weights, B);
fprintf('build hdr image complete\n')


%% tone manipulation
sigma_r = 2.5;
alpha = 0.3;
beta = 0.5;
numref = 100;
colorRemapping = 'lum';
domain = 'log';
R = lapfilter(I,sigma_r,alpha,beta,numref,colorRemapping,domain);
figure; clf; imshow(R);