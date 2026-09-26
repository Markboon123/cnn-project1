function outarray = apply_imnormalize(inarray)
%APPLY_IMNORMALIZE  CNN layer 1
%   outarray = apply_imnormalize(inarray)
%   inarray  : N x M x 3 uint8 RGB image
%   outarray : N x M x 3 double
%
%   Each channel is scaled to [0,1], then the CIFAR-10 channel mean is
%   subtracted and the result is divided by the channel standard deviation:
%       Out(:,:,k) = (In(:,:,k)/255 - mu(k)) / sigma(k)

% CIFAR-10 statistics for (R, G, B), hardcoded from the handout
mu    = [0.4914 0.4822 0.4465];
sigma = [0.2470 0.2435 0.2616];

% Convert to double BEFORE any arithmetic. uint8 math rounds and clips to
% 0..255, so subtracting the mean in uint8 would wipe out every negative value.
tmp = double(inarray) / 255.0;

outarray = zeros(size(tmp));
for k = 1:3                                    % R, G, B handled separately
    outarray(:,:,k) = (tmp(:,:,k) - mu(k)) / sigma(k);
end
end
