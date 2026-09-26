function outarray = apply_fullconnect(inarray, filterbank, biasvals)
%APPLY_FULLCONNECT  Fully connected layer.
%   outarray = apply_fullconnect(inarray, filterbank, biasvals)
%   inarray    : N x M x D1
%   filterbank : N x M x D1 x D2 (every filter is the size of the whole input)
%   biasvals   : vector of D2 bias values
%   outarray   : 1 x 1 x D2
%
%   Out(1,1,l) = sum_i sum_j sum_k F_l(i,j,k) * In(i,j,k)  +  b_l
%
%   Each output is one dot product between the input and the l-th filter,
%   plus the l-th bias.

D2 = size(filterbank, 4);

if ~isequal(size(filterbank(:,:,:,1)), size(inarray))
    error('apply_fullconnect: each filter must be the same size as the input');
end

% (:) unrolls an array into a column in the same (column-major) order for
% the input and for every filter, so element (i,j,k) of the filter always
% multiplies element (i,j,k) of the input.
x = inarray(:);

outarray = zeros(1, 1, D2);
for l = 1:D2
    Fl = filterbank(:,:,:,l);                       % l-th filter, N x M x D1
    outarray(1,1,l) = sum(Fl(:) .* x) + biasvals(l);   % triple sum + bias
end
end
