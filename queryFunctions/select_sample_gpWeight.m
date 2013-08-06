% Query a new sample (with maximum weight in the updated weight vector) within an active learning experiment according to the work:
%
% Alexander Freytag  and Paul Bodesheim and Erik Rodner and Joachim Denzler:
% "Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
% Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function selected_sample = select_sample_gpWeight(Ks_unlabeled, Kss_unlabeled, L, alpha, gpnoise)
%
% BRIEF:
%   Query a new sample from a pool of unlabeled ones such that the queried sample 
%   has the largest absolute weight after updating the alpha vector of the GP regression model.
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

function selected_sample = select_sample_gpWeight(Ks_unlabeled, Kss_unlabeled, L, alpha, gpnoise)
    mu = Ks_unlabeled'*alpha;
        
    sn2 = exp(2*gpnoise);
    V  = L'\Ks_unlabeled;
    var = max(0,Kss_unlabeled-sum(V.*V./(sn2))');
        
    % --- compute the first term ---  
    % which downweights the impact of a new sample by the amount of
    % predicted uncertainty
    % note: actually, the first term is the invers of this number, but we
    % call the rdivide method lateron for easier computation
    firstTerm = sqrt(var.^2+sn2^2);

    % --- compute the second term --- 
    % actually, for this query strategy the second term is constant to 1
    % therefore, we skip it since multiplying by one
    % doesn't change anything at all  

    % --- compute the third term
    % this is the difference between predicted label and GT label
    % since we have not GT info, we take the sign of the most plausible class
    % -> in the binary setting, this is simply the sign of the mean value
    thirdTerm=mu-sign(mu);

    % we are interested in overall changes, so we take the abs
    thirdTerm=abs(thirdTerm);

    % --- combine (multiply) all two (implicitely three) terms ---    
    % use the built-in function for element-wise division
    % where the vector is automatically extended in the second dimension
    % towards the right size
    %
    weight=bsxfun(@rdivide,thirdTerm,firstTerm);     
    
    % take the one with the largest resulting weight
    % that is, we chose the point which has the highest influence for its most
    % plausible class
    [~, selected_sample]=max(weight);
 end