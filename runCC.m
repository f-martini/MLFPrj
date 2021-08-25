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

% Functions path:
addpath(fullfile('./fun'));

% Defalut values:
defdatasetPath = fullfile('./data');
maxIter = 1000;
defH = 0.1;
defK = 5;
defT = 0;
flag = true;

%% Dataset Loading
datasetPath = input('\nSubmit dataset path:\n\n', 's');
if isempty(datasetPath)
    datasetPath = defdatasetPath;
end
imds = imageDatastore(datasetPath, 'IncludeSubfolders',true);


while flag
    %% 
    % Clustering Parameter  

    % Dictionary Learning:
    fprintf(['\nPlease, select a FEATURE EXTRACTION techinique submitting its number:\n' ...
            '0 - PCA: Eigenfaces\n' ...
            '1 - Bag of Words: Regular Grid\n\n']);

    reply = input('');
    if ~isempty(reply) & reply == 1
        featExtraction = @RG;
    else
        featExtraction = @PCA; %Default   
    end

    % Clustering Techinque;
    fprintf(['\nPlease, select a CLUSTERING techinique submitting its number:\n' ...
            '0 - BSAS\n' ...
            '1 - Manifold Learning\n' ...
            '2 - Expectation Maximization\n\n']);

    reply = input('');
    if isempty(reply)
        clustAlg = @EM; %Default
        prm = initParam(defK,'\nExpectation Maximization --- Submit K value:\n');
    else
        switch (reply)
            case 0
                clustAlg = @KM;
                prm = initParam(defT,'\nBSAS --- Submit threshold:\n');

            case 1
                clustAlg = @ML;
                prm = InitParam(defH, '\nManifold Learning --- Submit H value:\n');

            otherwise
                clustAlg = @EM; %Default
                prm = initParam(defK,'\nExpectation Maximization --- Submit K value:\n');
        end
    end


    % Result Saving
    reply = input('\nWould you like to save the results? [Y/N]\n\n');
    if ~isempty(reply) & reply == 'N'
        saveRes = reply;
    else
        saveRes = 'Y'; %Default
    end  

    %%
    % Feature Extraction:
    
    resfile = ['./save/', func2str(featExtraction), 'res', '32', '.mat'];
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

    C = clustAlg(F);

    %%
    % Exit flag update
    reply = input('\nWould you like to try with other parameters? [Y/N]\n\n');
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
    if ~isempty(reply) & isnumeric(reply)
        K = reply;
    else
        K = default;
    end

end
      



