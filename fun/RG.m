function data = RG(imds)

% This function implements Regular Grid alghoritm for features extraction. 
% It extracts descriptors from each image of the dataset and computes a 
% dictionary by performing K-means alghoritm on the set of descriptors 
% with an hardcoded K = 64.
% The function returns a matrix of size N x 64 with N = cardinality of 
% the dataset.
% The algorithm will compute patches of size 32x32.

M = size(imds.Files, 1);
r = 256;
c = 256;
ks = 32;
ptcs = M*((r*c)/(ks*ks));
A = zeros(ks*ks, ptcs);
B = zeros(ptcs, 1);
T = 32 - 1;

cnt = 1;
for i=1:M
    img = imresize(rgb2gray(readimage(imds,i)), [r c]);
    for x=1:ks:r
        for y=1:ks:c
           tmp = img(x:x+ks-1, y:y+ks-1); 
           if (nnz(tmp) > ks/2)
                B(cnt) = 1; % trace of the 'interesting' patches   
                A(:, cnt) = tmp(:);
           end
           cnt = cnt + 1;
        end
    end  
end

% Dictionary Learning
% K-means initialization
C = zeros(nnz(B), 1);
cnt = 1;
for i=1:size(B,1)
    if B(i)
        C(cnt) = i;
        cnt = cnt + 1;
    end    
end

k_o = randsample(C, T);
k_old = zeros(ks*ks, T);
k_old = double(k_old);

for i=1:T
    k_old(:,i) = A(:,k_o(i));
end

% K-means: matrix B will be reused in order to reduce resources usage 
flag = 1;
while 1
    sum = zeros(ks*ks+1, T);
    for i=1:size(C, 1)
        B(C(i)) = 1;
        min = norm(A(:,C(i))-k_old(:,1));
        for d=2:T
            tmp = norm(A(:,C(i))-k_old(:,d));
            if min > tmp
                min = tmp;
                B(C(i)) = d;
            end    
        end
        sum(1:ks*ks, B(C(i))) = sum(1:ks*ks, B(C(i))) + A(:,C(i));
        sum(ks*ks+1, B(C(i))) = sum(ks*ks+1, B(C(i))) + 1;
    end
    count = 0;
    for i=1:T
        tmp = sum(1:ks*ks, i) / sum(ks*ks+1, i);
        sub = tmp - k_old(:,i);
        if nnz(sub) > 0
            flag = 0;
            k_old(:, i) = tmp;
            count = count + 1;
        end
    end
    fprintf('%d miss convergence...\n', count);
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
            data(32, i) = data(32, i) + 1;
        else
            data(B(d*i+d+1), i) = data(B(d*i+d+1), i) + 1;
        end
    end
end


