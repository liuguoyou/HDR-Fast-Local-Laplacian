% Creates a list of all pictures and their exposure values in a certain directory.
%
% Note that the directory must only contain images wich are named according to the
% naming conventions, otherwise this function fails.
% 
% Filename naming conventions:
% The filename of a picture must contain two numbers specifying the
% exposure time of that picture. The first number specifies the nominator,
% the second one the denominator. E.g. "image_1_15.jpg" specifies that this
% image has been taken with an exposure of 1/15 second.
function [filenames, exposures, numExposures] = readDir(dirName)

    filelist = dir(dirName);
    numExposures = size(filelist, 1)-2;
    filenames = cell(numExposures, 1);
    exposures = zeros(numExposures, 1);
    
    for i = 3:size(filelist,1)
        filenames{i-2} = strcat(dirName,filelist(i).name);
    end
 
    for i = 1 : numExposures
        filename = filenames{i};
        [s, f] = regexp(filename, '(\d+)');
        nominator = filename(s(1):f(1));
        denominator = filename(s(2):f(2));
        exposure = str2double(nominator) / str2double(denominator);
        exposures(i) = exposure;
    end

    % sort ascending by exposure
    [exposures, indices] = sort(exposures);
    filenames = filenames(indices);

    % then inverse to get descending sort order
    exposures = exposures(end:-1:1);
    filenames = filenames(end:-1:1);