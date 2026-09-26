function outarray = apply_relu(inarray)
%APPLY_RELU  Rectified linear unit .
%   outarray = apply_relu(inarray)
%   inarray  : N x M x D
%   outarray : N x M x D, with Out(i,j,k) = max(In(i,j,k), 0)
%
%   max() aginst the scalar 0 is applied to every element, so all
%   negative values become 0 and everything else passes through unchanged.

outarray = max(inarray, 0);
end
