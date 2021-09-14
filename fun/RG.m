function data = RG(imds)

% This function implements Regular Grid alghoritm for features extraction. 
% It extracts descriptors from each image of the dataset and computes a 
% dictionary by performing K-means alghoritm on the set of descriptors 
% with an hardcoded K = 32.
% The function returns a matrix of size N x 32 with N = cardinality of 
% the dataset.
% The algorithm will compute patches of size 32x32.

M = size(imds.Files, 1);
r = 256;                            % resize row dim    
c = 256;                            % resize column dim
ks = 32;                            % patch size
ptcs = M*((r*c)/(ks*ks));           % patches count
A = zeros(ks*ks, ptcs);             % vectorize patches matrix
B = zeros(ptcs, 1);                 % 'interesting' patches logic vector
T = 32 - 1;                         % number of features

cnt = 1;
for i=1:M
    img = imresize(rgb2gray(readimage(imds,i)), [r c]);                 % dataset loading and resizing
    for x=1:ks:r
        for y=1:ks:c
           tmp = img(x:x+ks-1, y:y+ks-1); 
           if (nnz(tmp) > ks/2)
                B(cnt) = 1;                                             % trace of the 'interesting' patches   
                A(:, cnt) = tmp(:);                                     % A filling
           end
           cnt = cnt + 1;
        end
    end  
end


% Dictionary Learning
% K-means initialization
C = find(B);                            % list of valid indexes
nvld = size(C, 1);                       

k_o = randsample(C, T);                 % random choice among valid indexes
k_old = A(:,k_o);                       % initialization of the means matrix (1 mean = 1 word)


% K-means: matrix B will be reused as labels-vector in order to reduce resources usage 
flag = 1;
while 1
    summ = zeros(ks*ks, T);                                                     % contain the sum of the patches in a cluster 
    cl_c = zeros(1, T);                                                         % store the size of each cluster 
    
    for i=1:T
        dist = sqrt(bsxfun(@sum, bsxfun(@minus, A(:,C), k_old(:,i)).^2, 1));    % computes the distance between each patches wrt the i-th mean
        if i == 1
            old_dist = dist;                                                    % initialize old distances
            continue;
        end
        
        for d=1:nvld
            if dist(d) < old_dist(d)                                            % if the new distance computed is smaller than the previous...
                B(C(d)) = i;                                                    % update cluster associated with the C(d)-th patch                             
                old_dist(d) = dist(d);                                          % update old distance
            end    
        end
        
    end
    
    for i=1:size(C, 1)
        summ(:, B(C(i))) = summ(:, B(C(i))) + A(:,C(i));                        % sum each patch of the same cluster
        cl_c(B(C(i))) = cl_c(B(C(i))) + 1;                                      % update the count
    end
    count = 0;
    
    for i=1:T
        tmp = summ(:, i) / cl_c(i);                      % compute new mean
        sub = tmp - k_old(:,i);
        if nnz(sub) > 0
            flag = 0;
            k_old(:, i) = tmp;                           % update mean
            count = count + 1;
        end
    end
    fprintf('%d miss convergence...\n', count);          % print number of clusters whose means has changed
    
    if flag
       break; 
    end  
    flag = 1;
  
end

% Image Embedding
% All the patches that has not been considered for the Dictionary Learning 
% will be associated with an additional 'all-zeros' word/patch.

data = zeros(32, M);
for i=1:M
    for d=0:((r*c)/(ks*ks))-1
        if B(d*i+d+1) == 0
            data(32, i) = data(32, i) + 1;                      % if the image contains not-interesting patches increment the last feature
        else
            data(B(d*i+d+1), i) = data(B(d*i+d+1), i) + 1;
        end
    end
end


