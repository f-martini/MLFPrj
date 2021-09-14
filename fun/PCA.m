function data = PCA(imds)

% This function apply PCA (eigenfaces) to the input imagedatastore object.
% It returns a matrix of size N x 32 with N = cardinality of the dataset.

M = size(imds.Files, 1);
r = 256;
c = 256;
A = zeros(r*c, M);
A = double(A);
T = 32;

for i=1:M
    tmp = imresize(rgb2gray(readimage(imds,i)), [r c]);         % dataset loading
    A(:, i) = tmp(:);
end

media = mean(A,2);                                              % mean computation

for i=1:M
    A(:,i) = A(:,i) - media;
end

[U,lambda] = eigen_training(A);
data = U(:,1:T)'*A;

%y = cumsum(lambda)/sum(lambda);
%plot(y); title('Modelled Information')
%scatter([1:M],y);



