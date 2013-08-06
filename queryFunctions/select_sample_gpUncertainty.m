% Query a new sample (with smallest "uncertainty", i.e.,  predictive mean / predictive variance) within an active learning experiment according to the work:
%
% Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler:
% "Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
% Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function selected_sample = select_sample_gpUncertainty(Ks_unlabeled, Kss_unlabeled, L, alpha, gpnoise)
%
% BRIEF:
%   Query a new sample from a pool of unlabeled ones such that the queried sample 
%   results in the smallest "uncertainty", which is predictive mean divided by predictive variance. 
%   This query strategy was introduced by Kapoor et al. in "Gaussian Processes
%   for Object Categorization" (2010, IJCV)
%
% INPUT: 
%   Ks_unlabeled            -- (n_t x n_u1) matrix of self similarities between n_t training samples and n_u unlabeled samples
%   Kss_unlabeled           -- (n_u x 1) vector of self similarities of n_u unlabeled samples
%   L                       -- (n_t x n_t) cholesky matrix (upper triangle) computed from the
%                              regularized kernel matrix of all training samples
%   alpha                   -- (n_t x 1) column vector, weight vector of GP
%                              regression model
%   gpnoise                 -- (1 x 1) scalar, indicating the noise for
%                              used for model regularization
%
% OUTPUT:
%   selected_sample         -- scalar, index of chosen sample
% 
% (C) copyright by Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler
%

function selected_sample = select_sample_gpUncertainty(Ks_unlabeled, Kss_unlabeled, L, alpha, gpnoise)
    mu = Ks_unlabeled'*alpha;
    
    sn2 = exp(2*gpnoise);
    V  = L'\Ks_unlabeled;
    var = max(0,Kss_unlabeled-sum(V.*V./(sn2))');
    
    heur = abs(mu)./sqrt(var+sn2);
    [~, selected_sample] = min(heur);
end