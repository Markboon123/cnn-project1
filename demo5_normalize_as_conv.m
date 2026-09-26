%% DEMO5_NORMALIZE_AS_CONV  Extra credit 2: imnormalize as a convolution layer.
%
%   Normalization computes, for each channel l = 1,2,3,
%       Out(:,:,l) = (In(:,:,l)/255 - mu_l) / sigma_l
%                  = (1/(255*sigma_l)) * In(:,:,l)  +  (-mu_l/sigma_l),
%   a per-channel scale plus a per-channel offset. A convolution layer computes
%       Out(:,:,l) = sum over k of F_l(:,:,k) * In(:,:,k)  +  b_l,
%   so it reproduces normalization exactly with
%       filterbank: size 1 x 1 x 3 x 3,  F_l(1,1,k) = 1/(255*sigma_l) if k == l, else 0
%       biases:     length 3,            b_l = -mu_l/sigma_l
%   A 1 x 1 filter touches no neighbours, so the 180-degree flip and the
%   zero padding never matter, and the zero weights for k ~= l keep the
%   color channels from mixing. The raw image must be converted to double
%   first (apply_convolve does this), otherwise the negative outputs are lost.

clear;
load('CNNparameters.mat');
load('debuggingTest.mat');          % imrgb, layerResults

mu    = [0.4914 0.4822 0.4465];
sigma = [0.2470 0.2435 0.2616];

normfilters = zeros(1, 1, 3, 3);    % R x C x D1 x D2 = 1 x 1 x 3 x 3
normbiases  = zeros(3, 1);
for l = 1:3
    normfilters(1, 1, l, l) = 1 / (255 * sigma(l));
    normbiases(l)           = -mu(l) / sigma(l);
end

fprintf('Filter weights W(k,l) = F_l(1,1,k)  (rows k = input channel, columns l = output channel):\n');
disp(squeeze(normfilters));
fprintf('Biases b = [%s]\n\n', strtrim(sprintf('%.5f ', normbiases)));

% same image through both versions of layer 1
viaConv = apply_convolve(double(imrgb), normfilters, normbiases);
viaNorm = apply_imnormalize(imrgb);
fprintf('layer 1 as convolution vs apply_imnormalize: max |difference| = %.1e\n', ...
        max(abs(viaConv(:) - viaNorm(:))));
fprintf('layer 1 as convolution vs debuggingTest.mat: max |difference| = %.1e\n', ...
        max(abs(viaConv(:) - layerResults{1}(:))));

% swap it into the network: layer 1 becomes an ordinary 'convolve' layer
convtypes = layertypes;    convtypes{1}   = 'convolve';
convbanks = filterbanks;   convbanks{1}   = normfilters;
convbias  = biasvectors;   convbias{1}    = normbiases;
results = cnn_forward(imrgb, convtypes, convbanks, convbias);
fprintf('final probabilities vs debuggingTest.mat:    max |difference| = %.1e\n', ...
        max(abs(results{end}(:) - layerResults{end}(:))));
