function outarray = apply_convolve(inarray, filterbank, biasvals)
%APPLY_CONVOLVE  Convolution layer: a bank of D2 filters plus D2 biases.
%   outarray = apply_convolve(inarray, filterbank, biasvals)
%   inarray    : N x M x D1
%   filterbank : R x C x D1 x D2, where filterbank(:,:,:,l) is the l-th filter
%   biasvals   : vector of D2 bias values
%   outarray   : N x M x D2
%
%   Out(:,:,l) = sum over k = 1..D1 of  F_l(:,:,k) * In(:,:,k)  +  b_l
%
%   * is true 2D convolution (imfilter 'conv', filter rotated 180 degrees),
%   not correlation; the provided filters only work with convolution.
%   'same' keeps each result N x M, and 0 zero-pads the off-image pixels.

N  = size(inarray, 1);
M  = size(inarray, 2);
D1 = size(inarray, 3);
D2 = size(filterbank, 4);

if size(filterbank, 3) ~= D1
    error('apply_convolve: filter depth (%d) does not match input depth (%d)', ...
          size(filterbank, 3), D1);
end

% imfilter returns the same class as its input, so make sure we filter doubles
inarray = double(inarray);

outarray = zeros(N, M, D2);
for l = 1:D2                    % one output channel per filter
    for k = 1:D1                % add up the 2D convolutions over the input channels
        outarray(:,:,l) = outarray(:,:,l) + ...
            imfilter(inarray(:,:,k), filterbank(:,:,k,l), 'conv', 'same', 0);
    end
    outarray(:,:,l) = outarray(:,:,l) + biasvals(l);   % then add the l-th bias
end
end
