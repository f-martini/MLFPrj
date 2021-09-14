function [model, flabels] = BSAS(X, K)

% This function is a basic implementaton of the BSAS algorithm.
% It takes as input the dataset X and K (maximimum number of clusters).
% The function tries to find a optimal number of clusters by changing the 
% threshold paramter used to determin whether or not generate a new cluster.

it = 20;                                        % hardcoded number of different threshold values
[f, N] = size(X);                                

% Model initialization: 
model.means = cell(it, K);                      % means of each cluster for each iteration

for i=1:it
    for d=1:K
        model.means{i, d}= zeros(f, 1);
    end
end

model.count = zeros(it, K);                     % size of each cluster for each different threshold
model.params = zeros(it, 2);                    % thresholds/clusters pairs used/generated in each iteration
 
flabels.labels = zeros(1, N);                   % cluster label for each element obtained wrt the best clustering parameters
tmplabels = zeros(1, N);                        % labels vector used during computation in order to keep in flabels.labels only the current best result

% Find the max distance between instances and initialize the threshold to its half 
thr = 0;

for i=1:N
    for d=1:N
        if i~=d
           dist = norm(X(:,i) - X(:,d));
           if thr < dist
               thr = dist;
           end
        end
    end
end

thr = thr/2;                                    % base threshold
model.params(1,1) = thr;                        % update model.params


% Clustering loop: at each iteration BSAS will be computed with a different threshold 
for d=1:it
    
    if d > 1
        model.params(d,1) = model.params(d-1,1) - thr/it;               % subtract 1/20 of the original threhsold to the last threshold each new iteration
    end
   
    model.means{d, 1}(:) = X(:, 1);                                     % initialize the mean of the first cluster to the first instance
    model.count(d, 1) = 1;                                              % initialize count for the first cluster
    tmplabels(1) = 1;                                                   % associate the first instance to the first cluster
    
    for i=2:N 
        min = norm(model.means{d, 1}(:) - X(:, i));                     % initialize the minimum distance of each instance to the distance between it and the first cluster mean 
        pos = 1;
        posz = 0;
        
        % Compute the distance between the current instance and the
        % existing cluster
        for k=2:K
            if model.count(d,k) == 0                                    
                posz = k;                                               % store the index of the new potential cluster
                break;
            end
            
            tmp = norm(model.means{d, k}(:) - X(:, i));                 % compute the distance between the other clusters meanns
            if tmp < min
                min = tmp;                                              % store the minimum distance
                pos = k;                                                % store the cluster label/index
            end
        end
        
        % Update the part of the model linked to the current parameters.
        % It's possible to update it instance per instance because we use
        % only the mean to describe the clusters models.
        if min < model.params(d,1) || posz == 0
            model.count(d, pos) = model.count(d, pos) + 1;                                                                          % increment the cluster count
            model.means{d, pos}(:) = (model.means{d, pos}(:) * (model.count(d, pos) - 1) + X(:, i)) / model.count(d, pos);          % compute new mean
            tmplabels(i) = pos;                                                                                                     % update labels vector    
        else
            model.count(d, posz) = model.count(d, posz) + 1;                                                                        % initialize new cluster    
            model.means{d, posz}(:) = X(:, i);                                                                                      % ...
            tmplabels(i) = posz;                                                                                                    % update labels vector
        end    
        
    end
    
    model.params(d,2) = nnz(model.count(d,:));                                              % store final number of cluster for this parameter
    
    % Update the final labels if variation wrt the mean of the distribution 
    % of the istances in the different cluster is the smallest.
    n = nnz(model.count(d,:));                                                               
    tmpvar = (sum(abs(model.count(d,1:n) - ones(1,n)* mean(model.count(d,1:n)))))/n;        % compute variation 
 
    if d == 1
        minvar = tmpvar;
        flabels.labels = tmplabels;
        final = d;
    else
        if tmpvar < minvar
            minvar = tmpvar;
            flabels.labels = tmplabels;
            final = d;
        end
    end
    
end

flabels.count = model.count(final,1:nnz(model.count(final,:)));                             % update count



