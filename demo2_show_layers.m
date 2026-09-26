%% DEMO2_SHOW_LAYERS  Look inside the CNN for one image.
%   Figure 1: the input image, the fully connected scores (layer 17) and the
%             softmax class probabilities (layer 18) as bar charts.
%   Figure 2: every channel of every image-shaped layer (layers 1-16) as a
%             greyscale image, one row per layer.
%   Figure 3: close-up of the channels of one layer (closeupLayer below).

clear;
load('CNNparameters.mat');                % layertypes, filterbanks, biasvectors
load('debuggingTest.mat', 'imrgb');
load('cifar10testdata.mat');              % imageset, trueclass, classlabels

% Image to look at: 0 = the debugging image, or an index 1..10000 into the test set
testIndex    = 0;
closeupLayer = 2;                         % layer shown in Figure 3

if testIndex == 0
    im = imrgb;
    imname = 'debugging image';
else
    im = imageset(:,:,:,testIndex);
    imname = sprintf('test image %d (%s)', testIndex, classlabels{trueclass(testIndex)});
end

layerResults = cnn_forward(im, layertypes, filterbanks, biasvectors);
scores = squeeze(layerResults{17});       % fully connected output, 10 x 1
probs  = squeeze(layerResults{18});       % softmax output, 10 x 1
[maxprob, maxclass] = max(probs);
fprintf('%s: estimated class is %s with probability %.4f\n', imname, classlabels{maxclass}, maxprob);

%% Figure 1: input -> scores -> probabilities
figure('Name', 'Input and output', 'Position', [100 100 1250 380]);
subplot(1, 3, 1);
image(im);
axis image off;
title(imname);

subplot(1, 3, 2);
bar(scores);
set(gca, 'XTick', 1:10, 'XTickLabel', classlabels);
xtickangle(45);
ylabel('score');
title('layer 17: fully connected scores');

subplot(1, 3, 3);
bar(probs);
hold on;
bar(maxclass, probs(maxclass), 'r');      % highlight the winning class
hold off;
set(gca, 'XTick', 1:10, 'XTickLabel', classlabels);
xtickangle(45);
ylim([0 1]);
ylabel('probability');
title(sprintf('layer 18: softmax, %s (%.2f)', classlabels{maxclass}, maxprob));

%% Figure 2: every channel of every image-shaped layer
show_all_layers(layerResults, layertypes);

%% Figure 3: close-up of one layer
show_channels(layerResults{closeupLayer}, ...
              sprintf('layer %d (%s)', closeupLayer, layertypes{closeupLayer}));
