function outarray = apply_maxpool(inarray)
%APPLY_MAXPOOL  2x2 max pooling over non-overlapping blocks.
%   outarray = apply_maxpool(inarray)
%   inarray  : 2N x 2M x D
%   outarray : N x M x D (half the rows and columns, same number of channels)
%
%   Implemented as in Figure 1B of the handout: split the image into four
%   subsampled images that take every other pixel starting at (1,1), (1,2),
%   (2,1) and (2,2). Each 2x2 block puts exactly one of its pixels in each
%   of the four images, so a pixelwise max over the four images equals the
%   max over each block. The ':' in the third index pools every channel
%   separately in the same step.

topLeft     = inarray(1:2:end, 1:2:end, :);   % start at (1,1)
topRight    = inarray(1:2:end, 2:2:end, :);   % start at (1,2)
bottomLeft  = inarray(2:2:end, 1:2:end, :);   % start at (2,1)
bottomRight = inarray(2:2:end, 2:2:end, :);   % start at (2,2)

% max() only compares two arrays at a time, so combine them in pairs
outarray = max(max(topLeft, topRight), max(bottomLeft, bottomRight));
end
