% Query a new sample (with minimum absolute predictive mean) within an active learning experiment according to the work:
%
% Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler:
% "Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
% Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function selected_sample = select_sample_gpMean(Ks_unlabeled, ~, ~, alpha, ~)
%
% BRIEF:
%   Query a new sample from a pool of unlabeled ones such that the queried sample 
%   has the smallest absolute predictive mean among all possible samples.
%   This query strategy was introduced by Kapoor et al. in "Gaussian Processes
%   for Object Categorization" (2010, IJCV)
%
% INPUT: 
%   Ks_unlabeled            -- (n_t x n_u1) matrix of self similarities between n_t training samples and n_u unlabeled samples
%   alpha                   -- (n_t x 1) column vector, weight vector of GP
%                              regression model
%
% OUTPUT:
%   selected_sample         -- scalar, index of chosen sample
% 
% (C) copyright by Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler
%

function selected_sample = select_sample_gpMean(Ks_unlabeled, ~, ~, alpha, ~)
    mu = Ks_unlabeled'*alpha;
    [~, selected_sample] = min(abs(mu));
end