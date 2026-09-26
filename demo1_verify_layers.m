%% DEMO1_VERIFY_LAYERS  Show that each layer function computes the right thing.
%   Part 1: small examples that can be checked by hand (handout + lecture).
%   Part 2: each layer run on its own, fed the EXPECTED output of the
%           previous layer from debuggingTest.mat, so an error would show
%           up in the layer that causes it instead of propagating.
%   Part 3: the whole 18-layer network run end-to-end from imrgb.
%   Part 4: why the handout insists on convolution, not correlation.
%
%   Needs CNNparameters.mat, debuggingTest.mat and cifar10testdata.mat in
%   the current folder.

clear;
load('CNNparameters.mat');                    % layertypes, filterbanks, biasvectors
load('debuggingTest.mat');                    % imrgb, layerResults
load('cifar10testdata.mat', 'classlabels');   % class names only
numLayers = length(layertypes);
okText = {'FAIL', 'ok'};

%% Part 1: hand-checkable examples
fprintf('--- Part 1: small hand-checkable examples ---\n');

% handout: pixel (128,64,200) normalizes to about (0.043, -0.950, 1.290)
% (exact arithmetic gives 1.2913 for blue; the handout's last digit is rounded)
pixel = uint8(cat(3, 128, 64, 200));          % a 1 x 1 x 3 "image"
fprintf('imnormalize (128,64,200) = (%.3f, %.3f, %.3f)   handout: about (0.043, -0.950, 1.290)\n', ...
        apply_imnormalize(pixel));

% ReLU zeroes the negatives
fprintf('relu [-2 -0.5 0 1.5 3]   = [%s]   expected [0 0 0 1.5 3]\n', ...
        strtrim(sprintf('%g ', apply_relu([-2 -0.5 0 1.5 3]))));

% handout Figure 1: 4x4 -> 2x2
A = [1 1 2 4; 5 6 7 8; 3 2 1 0; 1 2 3 4];
P = apply_maxpool(A);
fprintf('maxpool of Figure 1      = [%g %g; %g %g]   expected [6 8; 3 4]\n', P');

% worked convolution example from lecture: the filter is rotated 180
% degrees, and off-image pixels are zero (one channel, one filter, no bias)
f = [2 0 1; 1 -1 0; 0 1 1];
h = [3 1 2 0; 1 4 1 2; 2 0 3 1; 1 2 1 3];
C = apply_convolve(h, f, 0);
fprintf('convolve worked example  : pixel (2,2) = %g (expected 9), corner (1,1) = %g (expected 6 with zero pad)\n', ...
        C(2,2), C(1,1));

% softmax example from lecture: [3.2 5.1 -1.7] -> [0.13 0.87 0.00]
fprintf('softmax [3.2 5.1 -1.7]   = [%s]   expected [0.13 0.87 0.00]\n', ...
        strtrim(sprintf('%.2f ', apply_softmax(cat(3, 3.2, 5.1, -1.7)))));

% subtracting alpha keeps large scores from overflowing exp()
big   = cat(3, 1000, 1001, 1002);
naive = exp(big) / sum(exp(big(:)));
fprintf('softmax [1000 1001 1002] = [%s]   without alpha: [%s]\n', ...
        strtrim(sprintf('%.3f ', apply_softmax(big))), strtrim(sprintf('%g ', naive)));

%% Part 2: every layer on its own
fprintf('\n--- Part 2: each layer alone (input = expected output of the previous layer) ---\n');
tol = 1e-10;
allOK = true;
for d = 1:numLayers
    if d == 1
        layerIn = imrgb;
    else
        layerIn = layerResults{d-1};
    end
    layerOut = run_layer(layerIn, layertypes{d}, filterbanks{d}, biasvectors{d});

    sizeOK = isequal(size(layerOut), size(layerResults{d}));
    if sizeOK
        err = max(abs(layerOut(:) - layerResults{d}(:)));
    else
        err = Inf;
    end
    ok = sizeOK && err < tol;
    allOK = allOK && ok;
    fprintf('layer %2d  %-11s  output %2d x %2d x %2d   max |error| = %.1e   %s\n', ...
            d, layertypes{d}, size(layerOut,1), size(layerOut,2), size(layerOut,3), ...
            err, okText{ok + 1});
end

%% Part 3: the whole network, end to end
fprintf('\n--- Part 3: whole network end-to-end from imrgb ---\n');
myResults = cnn_forward(imrgb, layertypes, filterbanks, biasvectors);
worst = 0;
for d = 1:numLayers
    worst = max(worst, max(abs(myResults{d}(:) - layerResults{d}(:))));
end
fprintf('largest error over all %d layers = %.1e\n', numLayers, worst);
classprobvec = squeeze(myResults{end});
[maxprob, maxclass] = max(classprobvec);
fprintf('estimated class is %s with probability %.4f\n', classlabels{maxclass}, maxprob);

if allOK && worst < tol
    fprintf('\nAll layers match debuggingTest.mat.\n');
else
    fprintf('\nWARNING: some layers do not match debuggingTest.mat.\n');
end

%% Part 4: convolution vs. correlation
fprintf('\n--- Part 4: what if we used correlation instead? ---\n');
% Correlation is convolution with the filter rotated 180 degrees, so
% flipping the rows and columns of every conv filter gives the correlation version.
corrbanks = filterbanks;
for d = 1:numLayers
    if strcmp(layertypes{d}, 'convolve')
        corrbanks{d} = filterbanks{d}(end:-1:1, end:-1:1, :, :);
    end
end
corrLayer2 = apply_convolve(layerResults{1}, corrbanks{2}, biasvectors{2});
fprintf('layer 2 with correlation: max |error| = %.2f  (clearly wrong)\n', ...
        max(abs(corrLayer2(:) - layerResults{2}(:))));
corrResults = cnn_forward(imrgb, layertypes, corrbanks, biasvectors);
[corrprob, corrclass] = max(squeeze(corrResults{end}));
fprintf('whole network with correlation: %s with probability %.4f\n', ...
        classlabels{corrclass}, corrprob);
