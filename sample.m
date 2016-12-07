function [ red, green, blue ] = sample( image, sampleIndices )
    % Takes relevant samples of the input image
 
    redChannel = image(:,:,1);
    red = redChannel(sampleIndices);
    
    greenChannel = image(:,:,2);
    green = greenChannel(sampleIndices);
    
    blueChannel = image(:,:,3);
    blue = blueChannel(sampleIndices);
    
    
    
    
    
    
    
    


