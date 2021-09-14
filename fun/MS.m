function [model, data2cluster] = MS(dataPts, bandWidth);
%perform MeanShift Clustering of data using a flat kernel
%
% ---INPUT---
% dataPts           - input data, (numDim x numPts)
% bandWidth         - is bandwidth parameter (scalar)
% ---OUTPUT---
% model.clustCent       - is locations of cluster centers (numDim x numClust)
% data2cluster.labels   - for every data point which cluster it belongs to (numPts)
% data2cluster.count    - number of element for every cluster
% 
% Bryan Feldman 02/24/06
% MeanShift first appears in
% K. Funkunaga and L.D. Hosteler, "The Estimation of the Gradient of a
% Density Function, with Applications in Pattern Recognition"
%
% Adaptation of the following MathWorks:
%   " Bart Finkston (2021). Mean Shift Clustering 
%     (https://www.mathworks.com/matlabcentral/fileexchange/10161-mean-shift-clustering), 
%     MATLAB Central File Exchange. Retrieved August 29, 2021. "


%**** Initialize stuff ***
x = 1;                                                 % added parameter
[numDim,numPts] = size(dataPts);
numClust        = 0;
bandSq          = bandWidth^2;
initPtInds      = 1:numPts;
maxPos          = max(dataPts,[],2);                    % biggest size in each dimension
minPos          = min(dataPts,[],2);                    % smallest size in each dimension
boundBox        = maxPos-minPos;                        % bounding box size
sizeSpace       = norm(boundBox);                       % indicator of size of data space
stopThresh      = 1e-3*bandWidth;                       % when mean has converged
clustCent       = [];                                   % center of clust
beenVisitedFlag = zeros(1,numPts,'uint8');              % track if a points been seen already
numInitPts      = numPts;                               % number of points to possibly use as initilization points
clusterVotes    = zeros(1,numPts,'uint16');             % used to resolve conflicts on cluster membership


while numInitPts

    tempInd         = ceil( (numInitPts-1e-6)*rand);        % pick a random seed point
    stInd           = initPtInds(tempInd);                  % use this point as start of mean
    myMean          = dataPts(:,stInd);                     % intilize mean to this points location
    myMembers       = [];                                   % points that will get added to this cluster                          
    thisClusterVotes    = zeros(1,numPts,'uint16');         % used to resolve conflicts on cluster membership

    while 1     % loop until convergence
        
        sqDistToAll = sum((repmat(myMean,1,numPts) - dataPts).^2);      % dist squared from mean to all points still active
        inInds      = find(sqDistToAll < bandSq);                       % points within bandWidth
        thisClusterVotes(inInds) = thisClusterVotes(inInds)+1;          % add a vote for all the in points belonging to this cluster
        
        
        myOldMean   = myMean;                                   % save the old mean
        myMean      = mean(dataPts(:,inInds),2);                % compute the new mean
        myMembers   = [myMembers inInds];                       % add any point within bandWidth to the cluster
        beenVisitedFlag(myMembers) = 1;                         % mark that these points have been visited
        
        %**** if mean doesn't move much stop this cluster ***
        if norm(myMean-myOldMean) < stopThresh
            
            %check for merge posibilities
            mergeWith = 0;
            for cN = 1:numClust
                distToOther = norm(myMean-clustCent(:,cN));     % distance from posible new clust max to old clust max
                if distToOther < bandWidth/x                    % if its within bandwidth/x merge new and old
                    mergeWith = cN;
                    break;
                end
            end
            
            
            if mergeWith > 0    % something to merge
                c1 = sum(thisClusterVotes);                                 % new element count
                c2 = sum(clusterVotes(mergeWith,:));                        % element in the existing cluster
                clustCent(:,mergeWith)       = (c1 * myMean + c2 * clustCent(:,mergeWith))/(c1 + c2);       % record the max as the mean of the two merged 
                clusterVotes(mergeWith,:)    = clusterVotes(mergeWith,:) + thisClusterVotes;                % add these votes to the merged cluster
            else    %its a new cluster
                numClust                    = numClust+1;                   % increment clusters
                clustCent(:,numClust)       = myMean;                       % record the mean  
                clusterVotes(numClust,:)    = thisClusterVotes;
            end

            break;
        end

    end
    
    initPtInds      = find(beenVisitedFlag == 0);           % we can initialize with any of the points not yet visited
    numInitPts      = length(initPtInds);                   % number of active points in set
    fprintf('There are still %d active points.\n', numInitPts);
    
end

[val,data2cluster.labels] = max(clusterVotes,[],1);                % a point belongs to the cluster with the most votes
model.center = clustCent; 
data2cluster.count = zeros(1, numClust);

for i=1:numPts
    data2cluster.count(data2cluster.labels(i)) = data2cluster.count(data2cluster.labels(i)) + 1;
end
