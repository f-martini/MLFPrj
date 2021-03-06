function [U,lambda]=eigen_training(A)

% This function computes the eigenvector from the matrix A which contains 
% the input dataset (vectorized and peprocessed).  
% It computes the eigenfaces wrt the input dataset images and then extract 
% from them the proper M normalized eigenvectors. 

M = size(A,2);
N = size(A,1);
L = A'*A;

% computing eigenvalues of A'A
[vec,val] = eig(L);
val = diag(val);
[lambda, ind] = sort(val,'descend');
vec = vec(:,ind);

%getting back the M eigenvectors of AA'
U = A*vec;

%In order to be sure to have vectors of norm 1 
for i=1:M
    U(:,i)=U(:,i)/norm(U(:,i));
end

