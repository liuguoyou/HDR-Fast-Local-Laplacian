% Takes relevant samples from the images for use in gsolve.m
%
%
function [ zRed, zGreen, zBlue, sampleIndices ] = makeImageMatrix( filenames, numPixels )
    
    % determine the number of differently exposed images
    numExposures = size(filenames,1);
    
    
    % Create the vector of sample indices    
    % We need N(P-1) > (Zmax - Zmin)
    % Assuming the maximum (Zmax - Zmin) = 255, 
    % N = (255 * 2) / (P-1) clearly fulfills this requirement
    numSamples = ceil(255*2 / (numExposures - 1));
    
    % create a random sampling matrix, telling us which
    % pixels of the original image we want to sample
    % using ceil fits the indices into the range [1,numPixels+1],
    % i.e. exactly the range of indices of zInput
    step = numPixels / numSamples;
    % different from paper, which says the pixels are best sampled from regions 
    % of the image with low intensity variance
    sampleIndices = floor((1:step:numPixels));
    sampleIndices = sampleIndices';
    
    
    % allocate resulting matrices
    zRed = zeros(numSamples, numExposures);
    zGreen = zeros(numSamples, numExposures);
    zBlue = zeros(numSamples, numExposures);
    
    for i=1:numExposures
        
        % read the nth image
        image = imread(filenames{i});
        
        % sample the image for each color channel
        [zRedTemp, zGreenTemp, zBlueTemp] = sample(image, sampleIndices);
        
        % build the resulting, small image consisting
        % of samples of the original image
        zRed(:,i) = zRedTemp;
        zGreen(:,i) = zGreenTemp;
        zBlue(:,i) = zBlueTemp;
    end