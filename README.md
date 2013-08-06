COPYRIGHT
=========

This package contains Matlab source code for active learning experiments with Gaussian process regression models as described in:

Alexander Freytag  and Erik Rodner and Paul Bodesheim and Joachim Denzler:
"Labeling examples that matter: Relevance-Based Active Learning with Gaussian Processes".
Proceedings of the German Conference on Pattern Recognition (GCPR), 2013.

Please cite that paper if you are using this code!

(LGPL) copyright by Alexander Freytag and Erik Rodner and Paul Bodesheim and Joachim Denzler


CONTENT
=======

initWorkspace.m
demo.m
active_learning_gpml.m
queryFunctions/select_sample_random.m
queryFunctions/select_sample_gpMean.m
queryFunctions/select_sample_gpVariance.m
queryFunctions/select_sample_gpUncertainty.m
queryFunctions/select_sample_gpWeight.m
queryFunctions/select_sample_gpImpact.m
roc-analysis/roc.m
roc-analysis/auroc.m
README.txt
License.txt


USAGE
=====

- getting to know:      - run demo.m, which performs an AL experiment on synthetically created 5D-histograms. Change the settings
                          in the demo file to figure out how everything works. Don't forget to set up the gpml directory accordingly!

- run an AL experiment: - Use the function "active_learning_gpml" to run a comparison of different AL-strategies for a given scenario
                        - Please refer to the documentation in active_learning_gpml.m for explanations of input and output variables.



NOTE
====

To keep things simple, GP models are computed using the GPML toolbox:

@MISC{Rasmussen10:GPML,
  author = {C. E. {Rasmussen} and H. {Nickisch}},
  title = {GPML Gaussian Processes for Machine Learning Toolbox},
  year = {2010},
  note = {\url{http://mloss.org/software/view/263/}},
}

For computing kernel values needed within the experiments, you can rely on the GPML toolbox as well.

