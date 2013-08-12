function K = covmin(hyp, x, z)
% See also COVFUNCTIONS.M.

if nargin<2, K = '0'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(x)>0;        % determine mode

ell = exp(0);
% ell = exp(hyp(1));                              % exponent for histogram entries
sf2 = exp(0);
% sf2 = exp(hyp(2));                              % factor for histogram entries

% precompute min kernel
if dg                                                               % vector kxx
  K = sum(x,2);
%    K = ones(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sf2*min_kernel(x'.^ell);
  else                                                   % cross covariances Kxz
    K = sf2*min_kernel(x'.^ell,z'.^ell);
  end
end

if nargin<4                                                        % covariances

else                                                               % derivatives
  error('not yet implemented')
end
