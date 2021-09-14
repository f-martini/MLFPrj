%% Main Script 
%
%  This script provides a simple console-menu which let one choose between 
%  different kind of feature-extraction/clustering combinations.
%  This menu makes also possible to save the results of the computation (if
%  needed).
%  
%  This script is also responsible for loading the dataset whose path must
%  be submitted beforehand.
%
%  The results will be eventually saved in './save' directory. 

clear all;
close all;
clc;


%%
% Functions path:
addpath(fullfile('./fun'));

% Defalut values:
defdatasetPath = fullfile('./data');
defH = 2;
defK = 10;
defT = 5.0;
flag = true;


%% Dataset Loading
datasetPath = input('\nSubmit dataset path:\n\n', 's');
if isempty(datasetPath)
    datasetPath = defdatasetPath;
end
imds = imageDatastore(datasetPath, 'IncludeSubfolders',true);

% Prepare dataset for visualization
dataset = zeros(256, 256, size(imds.Files, 1));
for i=1:size(imds.Files, 1)
    tmp = imresize(rgb2gray(readimage(imds,i)), [256 256]);
    dataset(:,:, i) = tmp(:,:);
end

[datasetPath, namedata, ext] = fileparts(datasetPath);
if size(namedata, 2) == 0
   [datasetPath, namedata, ext] = fileparts(datasetPath);
end  


%%
while flag
    %% 
    % Clustering Parameter  

    % Dictionary Learning:
    fprintf(['\nPlease, select a FEATURE EXTRACTION techinique submitting its number:\n' ...
            '0 - PCA: Eigenfaces\n' ...
            '1 - Bag of Words: Regular Grid\n\n']);

    reply = input('');
    if ~isempty(reply) && reply == 1
        featExtraction = @RG;
    else
        featExtraction = @PCA; %Default   
    end

    % Clustering Techinque;
    fprintf(['\nPlease, select a CLUSTERING techinique submitting its number:\n' ...
            '0 - BSAS\n' ...
            '1 - Mean Shift\n' ...
            '2 - Expectation Maximization\n\n']);

    reply = input('');
    if isempty(reply)
        clustAlg = @EM; %Default
        prm = initParam(defK,'\nExpectation Maximization --- Submit K value:\n');
    else
        switch (reply)
            case 0
                clustAlg = @BSAS;
                prm = initParam(defT,'\nBSAS --- Submit max number of clusters K:\n');

            case 1
                clustAlg = @MS;
                prm = initParam(defH, '\nMean Shift --- Submit H value:\n');

            otherwise
                clustAlg = @EM; %Default
                prm = initParam(defK,'\nExpectation Maximization --- Submit K value:\n');
        end
    end


    % Result Saving
    reply = input('\nWould you like to save the results? [Y/N]\n\n', 's');
    if ~isempty(reply) && reply == 'N'
        saveRes = reply;
    else
        saveRes = 'Y'; %Default
    end  

    
    %%
    % Feature Extraction:
      
    resfile = ['./save/', namedata, func2str(featExtraction), 'res', '32', '.mat'];
    if isfile(resfile)
        load(resfile);
    else
        F = featExtraction(imds);
        if saveRes == 'Y'
            save(resfile, 'F');
        end
    end   
    
    
    %%
    % Clustering:
    fresfile = ['./save/', namedata, func2str(clustAlg), 'res', func2str(featExtraction), '32','par', num2str(prm), '.mat'];
    if isfile(fresfile)
        load(fresfile);
    else
        %A = normalize(F);
        A = normalize(F,  'norm');
        [model, res] = clustAlg(A, prm);
        if saveRes == 'Y'
            save(fresfile, 'model', 'res');
        end
    end

    
    %%
    % Visualization:
    
    ncls = max(res.labels);
    clsimages = zeros(256, 256, ncls);
    f1 = figure;
    for i=1:size(imds.Files, 1)
        clsimages(:,:,res.labels(i)) = clsimages(:,:,res.labels(i)) + dataset(:,:,i);
    end
    
    if ncls >= 25
        ncls = 25;
        [res.count, I] = sort(res.count, 'descend');
    else
        I = [1:1:ncls];
    end
    
    x = ceil(sqrt(ncls));
    
    for i=1:ncls
        clsimages(:,:,I(i)) = clsimages(:,:,I(i))/res.count(I(i));
        subaxis(x,x,i,'SpacingVertical',0.01,'SpacingHorizontal',0.01, 'Padding', 0, 'MarginLeft',.01,'MarginRight',.01);
        imagesc(clsimages(:,:,I(i)));
        colormap gray;
        axis off;
    end

    
    %%
    % Exit flag update
    reply = input('\nWould you like to try with other parameters? [Y/N]\n\n', 's');
    if isempty(reply) || reply ~= 'Y'
        flag = false;
    end  
    
end

%%
%% Utils
%%
%%

function K = initParam(default, txt)

    % Checks input validity 
    fprintf(txt);
    reply = input('');
    if ~isempty(reply) && isnumeric(reply)
        K = reply;
    else
        K = default;
    end

end
      



