function C = min_kernel(a, b)

if nargin<1  || nargin>2 || nargout>1, error('Wrong number of arguments.'); end
bsx = exist('bsxfun','builtin');      % since Matlab R2007a 7.4.0 and Octave 3.0
if ~bsx, bsx = exist('bsxfun'); end      % bsxfun is not yes "builtin" in Octave
[D, n] = size(a);

if nargin==1                                                     % subtract mean
  b = a; m = n;
else
  [d, m] = size(b);
  if d ~= D, error('Error: column lengths must agree.'); end
end

if bsx                                               % compute squared distances
	C = zeros(n,m);
	% This code is working but still inefficient :(
	if ( n > m )
	  	for i=1:m
				C(:,i)=sum(bsxfun(@min,a,b(:,i)),1)';
		end
	else
		for j=1:n
				C(j,:)=sum(bsxfun(@min,a(:,j),b),1);
		end
	end

else
	error('not yet implemented...');
end
C = max(C,0);          % numerical noise can cause C to negative i.e. C > -1e-14
