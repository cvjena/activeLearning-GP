%% set up workspace and matlab path
initWorkspace

%%  sample some random histograms

dim = 5;
m = 3.0;
var = 1.0;

offsetSpecific = 2.0;
varSpecific = 2.0;
    

%sample some training data
nTrain = 2;
% we simulate histograms since we use HIK as similarity measure in many
% computer vision applications
labeled_X = abs ( m + var.*randn(2*nTrain,dim) );
labeled_X(1:nTrain,1) = abs ( labeled_X(1:nTrain,1) + offsetSpecific + varSpecific.*randn(nTrain,1) );
labeled_X(nTrain+1:2*nTrain,3) = abs ( labeled_X(nTrain+1:2*nTrain,3) + offsetSpecific + varSpecific.*randn(nTrain,1) );
labeled_X = bsxfun(@times, labeled_X, 1./(sum(labeled_X, 2)));

labeled_y = [ ones(nTrain,1); 2 * ones(nTrain,1) ];

%sample some data to query from
nUnlabeled = 100;
unlabeled_X = abs ( m + var.*randn(2*nUnlabeled,dim) );
unlabeled_X(1:nUnlabeled,1) = abs ( unlabeled_X(1:nUnlabeled,1) + offsetSpecific + varSpecific.*randn(nUnlabeled,1) );
unlabeled_X(nUnlabeled+1:2*nUnlabeled,3) = abs ( unlabeled_X(nUnlabeled+1:2*nUnlabeled,3) + offsetSpecific + varSpecific.*randn(nUnlabeled,1) );
unlabeled_X = bsxfun(@times, unlabeled_X, 1./(sum(unlabeled_X, 2)));

%simulate your user response for this experiment
unlabeled_y = [ ones(nUnlabeled,1); 2 * ones(nUnlabeled,1) ];

%sample some data to test on
nTest = 100;
test_X = abs ( m + var.*randn(2*nTest,dim) );
test_X(1:nTest,1) = abs ( test_X(1:nTest,1) + offsetSpecific + varSpecific.*randn(nTest,1) );
test_X(nTest+1:2*nTest,3) = abs ( test_X(nTest+1:2*nTest,3) + offsetSpecific + varSpecific.*randn(nTest,1) );
test_X = bsxfun(@times, test_X, 1./(sum(test_X, 2)));

test_y = [ ones(nTest,1); 2 * ones(nTest,1) ];

%% set up variables and stuff 

% index of positive class
positive_class = 1;

% how many images do we want to query
numQueries = 50;

% further settings related to the classifier used
settings.gpnoise = 0.1;
settings.gp_numRegressors = 0.25;
settings.binarySVM_outlierratio = 0.1;
settings.cov = {'covmin'};
settings.loghyper = [];
settings.balanced_classification = false;

% which strategies do we want to compare?
%chose a subset of the currently supported ones: {'random','gp-mean', 'gp-var', 'gp-unc', 'gp-weight', 'gp-impact'};
queryStrategies = {'gp-mean', 'gp-impact'};

% after how many steps shall we perform an optimization of hyperparameters?
% note that usually hyperparameters vary less heavily given new input data
% than your current model might do
optimizationStep = 5;

% print additional debug information?
verbose = false;


%% run the active learning experiment with your settings and data
[perf_values alTimes chosenIndices optimizedParameters]= active_learning_gpml ...
          (labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_X, test_y, positive_class, numQueries, settings, queryStrategies, optimizationStep, verbose);
      
      
      
 %% plot the results
 %specific color coding for the GCPR-AL paper
 c={'k-', 'b-',   'c-',  'b--',    'g-',     'm-'};
 linewidth=3;
 
 perfPlot=figure(1);
 hold on;
 for i=1:size(perf_values,1)
    plot( perf_values(i,:), c{i}, 'LineWidth', 2*linewidth, 'MarkerSize',8 );
 end
 leg1=legend(queryStrategies, 'Location', 'SouthEast','fontSize', 16,'LineWidth', 3);
 text_h=findobj(gca,'type','text');  
 set(text_h,'FontSize',14);
 set(gca, 'FontSize', 14);
 set(get(gca,'YLabel'), 'FontSize', 14);
 set(get(gca,'XLabel'), 'FontSize', 14);
 title('Performance comparison');
 xlabel('Number of queries');
 ylabel('AUC [%]');
 hold off;
 
 timePlot=figure(2);
 bar(  alTimes  );
 
 text_h=findobj(gca,'type','text');  
 set(text_h,'FontSize',14);
 set(gca, 'FontSize', 14);
 set(get(gca,'YLabel'), 'FontSize', 14);
 set(get(gca,'XLabel'), 'FontSize', 14); 
 title('Time needed');
 xlabel('Method');
 ylabel('Time for AL Exp [s]');
 hold off;

 
 paramPlot=figure(3);
 hold on;
 for i=1:size(optimizedParameters,3)
    plot( optimizedParameters(1,:,i), c{i}, 'LineWidth', 2*linewidth, 'MarkerSize',8 );
 end
 leg3=legend(queryStrategies, 'Location', 'SouthEast','fontSize', 16,'LineWidth', 3);
  text_h=findobj(gca,'type','text');  
 set(text_h,'FontSize',14);
 set(gca, 'FontSize', 14);
 set(get(gca,'YLabel'), 'FontSize', 14);
 set(get(gca,'XLabel'), 'FontSize', 14);
 title('Parameter development');
 xlabel('Number of queries');
 ylabel('Noise Paramter');
 hold off;

 pause;
 close( perfPlot );
 close( timePlot );
 close( paramPlot );