function outarray = apply_softmax(inarray)
%APPLY_SOFTMAX  Convert a vector of scores into class probabilities.
%   outarray = apply_softmax(inarray)
%   inarray  : 1 x 1 x D
%   outarray : 1 x 1 x D, every value in [0,1], all values summing to 1
%
%   Out(1,1,k) = exp(In(1,1,k) - alpha) / sum over j of exp(In(1,1,j) - alpha)
%   with alpha = max over k of In(1,1,k).
%
%   Subtracting alpha does not change the result (the factor exp(-alpha)
%   cancels between numerator and denominator), but it makes the largest
%   exponent 0, so exp() cannot overflow to Inf for large scores.

alpha   = max(inarray(:));
expvals = exp(inarray - alpha);          % still 1 x 1 x D
outarray = expvals / sum(expvals(:));
end
