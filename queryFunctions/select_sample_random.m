% Randomly query a new sample within an active learning experiment according to the work:
%
% Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler:
% "Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
% Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function selected_sample = select_sample_random(~, Kss_unlabeled, ~, ~, ~)
%
% BRIEF:
%   Randomly query a new sample from a pool of unlabeled ones.
%
% INPUT: 
%   Kss_unlabeled           -- (n_u x 1) vector of self similarities of n_u unlabeled samples
%
% OUTPUT:
%   selected_sample         -- scalar, index of chosen sample
% 
% (C) copyright by Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler
%

function selected_sample = select_sample_random(~, Kss_unlabeled, ~, ~, ~)
    numberOfUnlabeled = length(Kss_unlabeled);
    selected_sample = randi(numberOfUnlabeled);
end