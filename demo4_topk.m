%% DEMO4_TOPK  Extra credit 1: top-k classification accuracy for k = 1..10.
%   y(k) = percentage of test images whose correct class is among the k
%   classes with the highest probability. y(1) is the usual accuracy and
%   y(10) is 100%. Uses the probabilities saved by demo3_evaluate.m.

clear;
if ~exist('cnn_results.mat', 'file')
    error('cnn_results.mat not found: run demo3_evaluate first.');
end
load('cnn_results.mat', 'classprobs');
load('cifar10testdata.mat', 'trueclass');
trueclass = double(trueclass(:));
[numImages, numClasses] = size(classprobs);

% Sort each image's classes from most to least probable, then find the
% position of the correct class in that ranking (1 = the CNN's first choice).
[~, ranking] = sort(classprobs, 2, 'descend');
rankOfTrue = zeros(numImages, 1);
for n = 1:numImages
    rankOfTrue(n) = find(ranking(n, :) == trueclass(n));
end

topk = zeros(1, numClasses);
for k = 1:numClasses
    topk(k) = 100 * mean(rankOfTrue <= k);    % correct class within the top k
end

fprintf('Top-k accuracy on %d test images:\n', numImages);
for k = 1:numClasses
    fprintf('  k = %2d : %6.2f%%\n', k, topk(k));
end

figure('Name', 'Top-k accuracy');
plot(1:numClasses, topk, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
for k = 1:numClasses
    text(k + 0.15, topk(k) - 4, sprintf('%.1f', topk(k)));
end
axis([1 numClasses 0 100]);
set(gca, 'XTick', 1:numClasses);
grid on;
xlabel('k (number of top-ranked classes allowed)');
ylabel('correct class in the top k (%)');
title('Top-k classification accuracy');
