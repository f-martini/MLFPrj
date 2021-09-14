function [model, labels] = EM(X, init)

% Perform EM algorithm for fitting the Gaussian mixture model.
% Input:  
%   X:      d x n data matrix
%   init:   number of cluster to find
% Output:
%   model.weight:   means of likelihood of the data wrt each cluster (prior)
%        .Sigma:    covariance matrices
%        .mu:       means vector
%        .R:        matrix IxJ which stores P(C_j|x_i) 
%        .llh:      loglikelihood
%   labels.labels:      vector of the cluster label assocated to each instance 
%         .count:       size of each cluster


%% initialization
%fprintf('EM for Gaussian mixture: running ... \n');
[d, n] = size(X);
R = initialization(X,init);
[thrash,label(1,:)] = max(R,[],2);
R = R(:,unique(label));

tol = 1e-10;                    % factor which determine convergence (wrt llh changes)
maxiter = 500;
llh = -inf(1,maxiter);          % log-likelihood at each step of the cycle
converged = false;
t = 1;
while ~converged && t < maxiter
    t = t+1;
    model = maximization(X,R);                      % maximization step
    [R, llh(t)] = expectation(X,model);             % expectation step
    converged = llh(t)-llh(t-1) < tol*abs(llh(t));  % check convergence
end
llh = llh(2:t);                                     % cut the first -Inf value


% Wrap the output model and labels
model.llh = llh;
model.R = R;
labels.count = zeros(1, init);
[trs, labels.labels] = max(R, [], 2);                                       % compute labels choosing the cluster with higher probability wrt each instance

for i=1:n
    labels.count(labels.labels(i)) = labels.count(labels.labels(i)) + 1;    % compute clusters sizes
end

function R = initialization(X, init)
[d,n] = size(X);
% random initialization
k = init; 
idx = randsample(n,k);                                          % extracts k random indexes 
m = X(:,idx);                                                   % extracts the corresponding points
[thrash,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);   % compute label for each instance of the dataset wrt the point randomly chosen
[u,thrash,label] = unique(label);       
while k ~= length(u)                                            % redo until k unique label has been found    
    idx = randsample(n,k);
    m = X(:,idx);
    [thrash,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,thrash,label] = unique(label);
end
    
R = full(sparse(1:n,label,1,n,k,n));                            % initialize P(C_j|x_i) to be 1 wrt the instances labels

function [R, llh] = expectation(X, model)
mu = model.mu;
Sigma = model.Sigma;
w = model.weight;                                       

n = size(X,2);
k = size(mu,2);
logRho = zeros(n,k);                                            

for i = 1:k
    logRho(:,i) = loggausspdf(X,mu(:,i),Sigma(:,:,i));          % P(xi|Cj) expressed as a multivariate normal probability density function
end
logRho = bsxfun(@plus,logRho,log(w));                           % sum over j logP(xi|Cj)+logP(Cj)
T = logsumexp(logRho,2);                                        % compute log(sum(exp())) -> sum_j(log(P(Cj|xi)P(Cj)))
llh = sum(T)/n;                                                 % mean of T
logR = bsxfun(@minus,logRho,T);                                 % logP(Cj|xi)
R = exp(logR);                                                  % P(Cj|xi)


function model = maximization(X, R)
[d,n] = size(X);
k = size(R,2);

nk = sum(R,1);
w = nk/n;                                                       % prior
mu = bsxfun(@times, X*R, 1./nk);                                % weighted means (by P(Cj|xi))

Sigma = zeros(d,d,k);
sqrtR = sqrt(R);

% compute covariance matrices
for i = 1:k
    Xo = bsxfun(@minus,X,mu(:,i));                              % subtract mean
    Xo = bsxfun(@times,Xo,sqrtR(:,i)');                         
    Sigma(:,:,i) = Xo*Xo'/nk(i);                                % weighted covariance
    Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6);                  % add a prior for numerical stability
end

model.mu = mu;
model.Sigma = Sigma;
model.weight = w;

function y = loggausspdf(X, mu, Sigma)
y=log(mvnpdf(X', mu', Sigma'));
