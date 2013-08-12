% Main method for an active learning experiment according to the work:
%
% Alexander Freytag and Erik Rodner and Paul Bodesheim and Joachim Denzler:
% "Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
% Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function [perf_values times chosenIndices optimizedParameters]= active_learning_gpml(labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_X, test_y, positive_class, numQueries, settings, queryStrategies, optimizationStep, verbose)
%
% BRIEF:
%   Perform an active-learning experiment, i.e., initially train a GP model
%   from labeled training examples, query new samples from a pool of
%   unlabeled examples, and evaluate the performance after every query on a
%   disjoint test set
%
% INPUT: 
%   labeled_X           -- (n_t x d) matrix with n_t training samples (rows) consisting of d dimensions (columns)
%   labeled_y           -- (n_t x 1) column vector containing (multi-class) labels of n_t training samples
%   unlabeled_X         -- (n_u x d) matrix with n_u samples (rows) consisting of d dimensions (columns)
%   unlabeled_y         -- (n_u x 1) column vector containing (multi-class)
%                          labels of n_u samples used as unlabeled pool within an AL experiment
%   test_X              -- (n_t x d) matrix with n_t test samples (rows) consisting of d dimensions (columns)
%   test_y              -- (n_t x 1) column vector containing (multi-class) labels of n_t test samples
%   positive_class      -- (1 x 1) scalar indicating the multi-class label
%                          of the class to be used as positive one
%   numQueries          -- (1 x 1) scalar, how often do we want to query unlabeled examples in a single run?
%   settings            -- (n x 1) column vector containing (multi-class) labels of n training samples
%   queryStrategies     -- (1 x k) cell array, specify k query strategies
%                          that shall be compared
%                          Right now, the following strategies are
%                          supported:
%                            - random
%                            - gp-mean  (minimum absolute predictive mean)
%                            - gp-var   (maximum predictive variance)
%                            - gp-unc   (minimum "uncertainty")
%                            - gp-weight  (largest weight after update)
%                            - gp-impact  (largest model change after update)
%   optimizationStep    -- (1 x 1) scalar, after how many AL-iterations do we want to perform
%                          a re-optimization of current hyperparameters?
%   verbose             -- (1 x 1) scalar, give some interesting output?
%
% OUTPUT:
%   perf_values         -- the actual accuracies after every query step
%                         (length(queryStrategies) x  numQueries+1)
%   times               -- the computation times needed for every run
%                         (including training, computation of kernel values, and so on)
%                         ( length(queryStrategies) x 1 )
%   chosenIndices       -- the queried indices for every run
%                         (numQueries x  length(queryStrategies) )
%   optimizedParameters -- optional, if given, optimization is performed
%                         every optimizationStep steps
%                         if optimizationStep == 0: ( length(hyp.cov) + length(hyp.lik)) 
%                         else:  ( length(hyp.cov)+length(hyp.lik) x floor(numQueries/optimizationStep)+1 x length(queryStrategies))
%
%  NOTE:
%    This program uses the GPML-Toolbox provided by 
%
% 
% (C) copyright by Alexander Freytag and Erik Rodner and Paul Bodesheim and Joachim Denzler
%

function [perf_values times chosenIndices optimizedParameters]= active_learning_gpml(labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_X, test_y, positive_class, numQueries, settings, queryStrategies, optimizationStep, verbose)
% function [perf_values times chosenIndices optimizedParameters] = active_learning_gpml(labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_X, test_y, positive_class, numQueries, settings, queryStrategies, optimizationStep, verbose)


  %% are we running octave?
  b_isOctave = exist('OCTAVE_VERSION') ~= 0;

  %%

  % relevant settings
  gpnoise = settings.gpnoise;
  %convert to gpml-scale
  gpnoise = 0.5*log(gpnoise);
  cov = settings.cov;
  loghyper.cov = settings.loghyper;
  lik = @likGauss; %standard assumption of Gaussian distribution y~f
  inf = @infExact;
  Ncg = 50;   % number of conjugate gradient steps, same as in GPML toolbox
  loghyper.lik  = gpnoise;
  mean = @meanZero; % zero-mean assumption
  loghyper.mean=[];  

  
  if nargin < 12
      verbose = false;
  end
  if nargin < 11
      optimizationStep = 0;
  end  

  %convert multi-class labels to binary labels for the specified task
  current_labels = 2*(labeled_y==positive_class)-1;
  
  %optimization?
  %right now, we always perform an initial optimization!
  if Ncg==0
    hyp = loghyper;
  else
    hyp = minimize(loghyper,'gp', -Ncg, inf, mean, cov, lik, labeled_X, current_labels); % opt hypers
  end
  
  if ( optimizationStep == 0)
    optParams = [hyp.cov; hyp.lik];
  else
      %pre-allocate memory
      optParams = zeros(length(hyp.cov)+length(hyp.lik), floor(numQueries/optimizationStep)+1,length(queryStrategies));
      %store initially optimized parameters
      optParams(:,1,1:length(queryStrategies)) = repmat([hyp.cov; hyp.lik],1,length(queryStrategies) );
  end

  
  %training
  %[~, ~, post] = gp(hyp, inf, mean, cov, lik, labeled_X, current_labels);
  % for octave compatibility
  [dummy1, dummy2, post] = gp(hyp, inf, mean, cov, lik, labeled_X, current_labels);
  alpha=post.alpha;
  L = post.L;
  sW = post.sW;
  sn2 = exp(2*hyp.lik);    % noise variance of likGauss
  
  % init output variable
  perf_values = zeros( length(queryStrategies), numQueries+1 );
  times = zeros ( length(queryStrategies),1 );
  
  %copute similarities to test samples  
  Ks = feval(cov{:},hyp.cov, labeled_X, test_X);
  
  %initial test evaluation
  classification_mu = Ks'*alpha;
  [tp fp] = roc( (test_y == positive_class), classification_mu);
  perf_values(1:length(queryStrategies),1) = auroc(tp,fp);
  
  if (verbose)  
    sprintf('score (AuROC): %d',perf_values(1,1))
    disp('first model evaluation')  
  end
  
  initialClassifier.L = L;
  initialClassifier.alpha = alpha;
  initialClassifier.sn2 = sn2;
  initialClassifier.hyp = hyp;
  initialClassifier.Ks = Ks;
  
  
  chosenIndices = zeros( numQueries,length(queryStrategies) );
  %store the indices of the unlabeled samples, that have been already
  %queried
  ind_labeled = [];
  ind_unlabeled = 1:length(unlabeled_y);
  %indices for test sets are not needed, since the test set is not affected
  %during active learning experiments
  
  initialClassifier.ind_labeled = ind_labeled;
  initialClassifier.ind_unlabeled = ind_unlabeled;
  
  for stratIndex=1:length(queryStrategies)

      if (verbose)
          disp(sprintf('Method %d of %d: %s', stratIndex, length(queryStrategies), queryStrategies{stratIndex}));       
      end     
      
      %select the strategy we want to use for actively querying new samples
        switch queryStrategies{stratIndex}
          case 'random'
              activeSelectionMethod = @select_sample_random;
          case 'gp-mean'
              activeSelectionMethod = @select_sample_gpMean;
          case 'gp-var'
              activeSelectionMethod = @select_sample_gpVariance;
          case 'gp-unc'
              activeSelectionMethod = @select_sample_gpUncertainty;
          case 'gp-weight' 
              activeSelectionMethod = @select_sample_gpWeight;
          case 'gp-impact'
              activeSelectionMethod = @select_sample_gpImpact;
          otherwise
            error('wrong mode or not yet implemented');
        end      
      
      %reset the classifier to the initial version - thereby identical
      %initial conditions for different AL strategies are guaranteed 
      L = initialClassifier.L;
      alpha = initialClassifier.alpha;
      sn2 = initialClassifier.sn2;
      hyp = initialClassifier.hyp;   
      Ks = initialClassifier.Ks;

      ind_labeled = initialClassifier.ind_labeled;
      ind_unlabeled = initialClassifier.ind_unlabeled;  


      if ( b_isOctave )
        timeStamp=tic();
      else
        timeStamp=tic;
      end
      
      %start actively querying new samples
      if ( numQueries > 0)
        
        Ks_unlabeled = feval(cov{:}, hyp.cov, [labeled_X;unlabeled_X(ind_labeled,:)], unlabeled_X(ind_unlabeled,:) );
        Kss_unlabeled = feval(cov{:}, hyp.cov, unlabeled_X(ind_unlabeled,:), 'diag');

          for i=2:numQueries+1

            % select next sample based on specified strategy

            %note: selected_sample is only the reletive index between 1 and
            %the current size of unlabeled samples -> convert it to the
            %original index afterwards            
            selected_sample = activeSelectionMethod(Ks_unlabeled, Kss_unlabeled, L, alpha, hyp.lik);

            if (verbose)
                disp(sprintf('Query %d: sample of class %d (positive class = %d)',i-1,unlabeled_y(selected_sample),positive_class));
            end
            % update sets of labeled and unlabeled samples
            originalIndexSelectedSample = ind_unlabeled(selected_sample);
            ind_labeled = [ind_labeled; originalIndexSelectedSample];
            ind_unlabeled(selected_sample) = [];
            
            current_labels = 2*([labeled_y; unlabeled_y(ind_labeled)]==positive_class)-1;
                        
            %%% re-run optimization or just a simple update?
            if ( (Ncg~=0) && (mod(i, optimizationStep) == 0) )
                %re-run optimization
                hyp = minimize(loghyper,'gp', -Ncg, inf, mean, cov, lik, [labeled_X;unlabeled_X(ind_labeled,:)], current_labels); % opt hypers
                
                optParams(:,floor(i/optimizationStep)+1,stratIndex) = [hyp.cov; hyp.lik];
  
                %re-train with newly optimized hyperparameters
                %[~, ~, post] = gp(hyp, inf, mean, cov, lik, [labeled_X;unlabeled_X(ind_labeled,:)], current_labels);
                % for octave compatibility
                [dummy1, dummy2, post] = gp(hyp, inf, mean, cov, lik, [labeled_X;unlabeled_X(ind_labeled,:)], current_labels);
                alpha=post.alpha;
                L = post.L;
                sn2 = exp(2*hyp.lik);    % noise variance of likGauss
                                
                %do we have to re-compute the kernel values?
                if (size(hyp.cov) == 0)
                    %just update kernel values
                    
                    % update similarities of unlabeled samples
                    Ks_unlabeled(:,selected_sample)=[];
                    Ks_unlabeled = [Ks_unlabeled; feval(cov{:},hyp.cov, unlabeled_X(ind_labeled(length(ind_labeled)),:), unlabeled_X(ind_unlabeled,:) )];

                    Kss_unlabeled(selected_sample) = [];

                    Ks = [Ks; feval(cov{:},hyp.cov, unlabeled_X(ind_labeled(length(ind_labeled)),:), test_X)];                      
                else
                    %re-compute kernel values using the new hyperparameters
                    
                    %copute similarities to test samples  
                    Ks = feval(cov{:},hyp.cov, [labeled_X;unlabeled_X(ind_labeled,:)], test_X);
                    %copute similarities to unlabeled samples  
                    Ks_unlabeled = feval(cov{:}, hyp.cov, [labeled_X;unlabeled_X(ind_labeled,:)], unlabeled_X(ind_unlabeled,:) );
                    Kss_unlabeled = feval(cov{:}, hyp.cov, unlabeled_X(ind_unlabeled,:), 'diag');
                end
                    
            else % no optimization
                % Cholesky update to include the selected sample
                l = (L'\ ( Ks_unlabeled(:,selected_sample)))/sn2;
                kss = (  feval(cov{:},hyp.cov,unlabeled_X(length(ind_labeled),:))  )/sn2;
                lss = sqrt(kss+1-l'*l);
                L = [L, l; zeros(1,length(l)), lss ];

                % update similarities of unlabeled samples
                Ks_unlabeled(:,selected_sample) = [];
                Ks_unlabeled = [Ks_unlabeled; feval(cov{:},hyp.cov, unlabeled_X(ind_labeled(length(ind_labeled)),:), unlabeled_X(ind_unlabeled,:))];
                
                Kss_unlabeled(selected_sample) = [];
                
                Ks = [Ks; feval(cov{:},hyp.cov, unlabeled_X(ind_labeled(length(ind_labeled)),:), test_X)];
            end


            % determine new alpha and update Ks for testing
            alpha = solve_chol(L,current_labels)/sn2;


            % evaluate current model
            classification_mu = Ks'*alpha;
            
            [tp, fp] = roc( (test_y==positive_class), classification_mu );
            perf_values(stratIndex,i) = auroc(tp,fp);

          end
      else
    %       disp('No queries to take, therefor we are done here.')
      end
         
      if ( b_isOctave )
        % OCTAVE compatibility
        times(stratIndex)=toc();           
      else
        times(stratIndex)=toc(timeStamp);
      end      

      chosenIndices(:,stratIndex) = ind_labeled;
      
  end %for-loop over query strategies

  if (nargin > 3)
      optimizedParameters = optParams;
  end  

end
