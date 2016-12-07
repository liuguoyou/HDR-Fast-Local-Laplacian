% Generates a hdr radiance map from a set of pictures
%
% parameters:
% filenames: a list of filenames containing the differently exposed
% pictures used to make a hdr from
% gRed: camera response function for the red color channel
% gGreen: camera response function for the green color channel
% gBlue: camera response function for the blue color channel
function [ hdr ] = hdr( filenames, gRed, gGreen, gBlue, w, dt )

    numExposures = size(filenames,1);
   
    % read the first image to get the width and height information
    image = imread(filenames{1});
    
    % pre-allocate resulting hdr image
    hdr = zeros(size(image));
    sum = zeros(size(image));
    
    for i=1:numExposures
        
        fprintf('Adding picture %i of %i \n', i, numExposures);

        image = double(imread(filenames{i}));

        wij = w(image + 1);        
        sum = sum + wij;
        
        m(:,:,1) = (gRed(image(:,:,1) + 1) - dt(1,i));
        m(:,:,2) = (gGreen(image(:,:,2) + 1) - dt(1,i));
        m(:,:,3) = (gBlue(image(:,:,3) + 1) - dt(1,i));
                
        % If a pixel is saturated, its information and
        % that gathered from all prior pictures with longer exposure times is unreliable. Thus
        % we ignore its influence on the weighted sum (influence of the
        % same pixel from prior pics with longer exposure time ignored as
        % well)
        % exposure picture should be read in from high ev to low ev
        
        saturatedPixels = ones(size(image));    
            
        saturatedPixelsRed = find(image(:,:,1) == 255);
        saturatedPixelsGreen = find(image(:,:,2) == 255);
        saturatedPixelsBlue = find(image(:,:,3) == 255);
            
        % Mark the saturated pixels from a certain channel in *all three*
        % channels
        dim = size(image,1) * size(image,2);
 
        saturatedPixels(saturatedPixelsRed) = 0;
        saturatedPixels(saturatedPixelsRed + dim) = 0;
        saturatedPixels(saturatedPixelsRed + 2*dim) = 0;
           
        saturatedPixels(saturatedPixelsGreen) = 0;
        saturatedPixels(saturatedPixelsGreen + dim) = 0;
        saturatedPixels(saturatedPixelsGreen + 2*dim) = 0;
            
        saturatedPixels(saturatedPixelsBlue) = 0;
        saturatedPixels(saturatedPixelsBlue + dim) = 0;
        saturatedPixels(saturatedPixelsBlue + 2*dim) = 0;

        % add the weighted sum of the current pic to the resulting hdr radiance map        
        hdr = hdr + (wij .* m);
        
        % remove saturated pixels from the radiance map and the sum (saturated pixels
        % are zero in the saturatedPixels matrix, all others are one)
        hdr = hdr .* saturatedPixels;
        sum = sum .* saturatedPixels;
    end
    
    
    % For those pixels that even in the picture with the smallest exposure time still are
    % saturated we approximate the radiance only from that picture instead
    % of taking the weighted sum
    saturatedPixelIndices = find(hdr == 0);
    
    % Don't multiply with the weights since they are zero for saturated
    % pixels. m contains the logRadiance value from the last pic, that one
    % with the smallest exposure time.
    hdr(saturatedPixelIndices) = m(saturatedPixelIndices);
    
    % Fix the sum for those pixels to avoid division by zero
    sum(saturatedPixelIndices) = 1;
    
    % normalize
    hdr = hdr ./ sum;
    hdr = exp(hdr);