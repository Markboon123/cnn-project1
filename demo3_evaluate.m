%% DEMO3_EVALUATE  Classify all 10000 CIFAR-10 test images and evaluate the CNN.
%   Builds the 10 x 10 confusion matrix A, where A(i,j) counts the test
%   images of true class i that the CNN classified as class j, and reports
%       accuracy = sum_i A(i,i) / sum_i sum_j A(i,j),
%   the per-class rates and the most frequent confusions. Saves the results
%   to cnn_results.mat (demo4_topk.m reuses the saved probabilities).
%
%   Takes a few minutes: every image needs 530 small imfilter calls.

clear;
load('CNNparameters.mat');      % layertypes, filterbanks, biasvectors
load('cifar10testdata.mat');    % imageset, trueclass, classlabels

numImages  = size(imageset, 4);
numClasses = length(classlabels);
trueclass  = double(trueclass(:));          % stored as uint8 in the .mat file

confusion  = zeros(numClasses, numClasses); % A(i,j) as described above
predclass  = zeros(numImages, 1);           % class chosen for every image
classprobs = zeros(numImages, numClasses);  % every image's probability vector

fprintf('Classifying %d test images...\n', numImages);
tic;
for n = 1:numImages
    layerResults = cnn_forward(imageset(:,:,:,n), layertypes, filterbanks, biasvectors);
    probs = squeeze(layerResults{end});       % 10 x 1 class probabilities

    % the class with the highest probability wins; if two were exactly
    % tied, max() returns the first one (the lower class index)
    [~, maxclass] = max(probs);
    predclass(n) = maxclass;
    classprobs(n, :) = probs';

    confusion(trueclass(n), maxclass) = confusion(trueclass(n), maxclass) + 1;

    if mod(n, 1000) == 0
        fprintf('  %5d / %d images  (%.0f s)\n', n, numImages, toc);
    end
end

%% Overall classification rate
numCorrect = sum(diag(confusion));
accuracy   = numCorrect / sum(confusion(:));
fprintf('\nClassification rate: %.2f%%  (%d of %d correct)\n', ...
        100*accuracy, numCorrect, sum(confusion(:)));

%% Confusion matrix as a table
shortnames = cell(1, numClasses);
for j = 1:numClasses
    shortnames{j} = classlabels{j}(1:min(5, end));
end
fprintf('\nConfusion matrix (rows = true class, columns = CNN output):\n');
fprintf('%12s', '');
fprintf('%7s', shortnames{:});
fprintf('\n');
for i = 1:numClasses
    fprintf('%12s', classlabels{i});
    fprintf('%7d', confusion(i, :));
    fprintf('\n');
end

%% Per-class results, easiest class first
rowsums  = sum(confusion, 2);      % test images per true class (1000 each)
colsums  = sum(confusion, 1)';     % how often the CNN output each class
perclass = diag(confusion) ./ rowsums;
[~, order] = sort(perclass, 'descend');
fprintf('\n%12s %9s %14s\n', 'class', 'correct', 'times output');
for i = order'
    fprintf('%12s %8.1f%% %14d\n', classlabels{i}, 100*perclass(i), colsums(i));
end

%% Most frequent confusions = largest off-diagonal entries
offdiag = confusion - diag(diag(confusion));
[counts, idx] = sort(offdiag(:), 'descend');
fprintf('\nMost frequent confusions:\n');
for t = 1:8
    [i, j] = ind2sub(size(offdiag), idx(t));
    fprintf('  %-10s classified as %-10s %4d times\n', classlabels{i}, classlabels{j}, counts(t));
end

%% Figure 1: confusion matrix
figure('Name', 'Confusion matrix', 'Position', [100 100 760 660]);
imagesc(confusion);
whiteToBlue = [linspace(1, 0.08, 256)', linspace(1, 0.27, 256)', linspace(1, 0.58, 256)'];
colormap(whiteToBlue);
colorbar;
axis image;
for i = 1:numClasses
    for j = 1:numClasses
        if confusion(i,j) > 0.5 * max(confusion(:))
            textColor = 'w';
        else
            textColor = 'k';
        end
        text(j, i, sprintf('%d', confusion(i,j)), ...
             'HorizontalAlignment', 'center', 'Color', textColor);
    end
end
set(gca, 'XTick', 1:numClasses, 'XTickLabel', classlabels, ...
         'YTick', 1:numClasses, 'YTickLabel', classlabels);
xtickangle(45);
xlabel('class output by the CNN');
ylabel('true class');
title(sprintf('Confusion matrix, %d test images: accuracy %.2f%%', numImages, 100*accuracy));

%% Figure 2: accuracy per class
figure('Name', 'Per-class accuracy');
bar(100 * perclass);
hold on;
plot([0.5, numClasses + 0.5], 100 * accuracy * [1 1], 'r--', 'LineWidth', 1.5);
hold off;
set(gca, 'XTick', 1:numClasses, 'XTickLabel', classlabels);
xtickangle(45);
ylim([0 100]);
ylabel('correctly classified (%)');
legend('per class', sprintf('overall %.2f%%', 100 * accuracy), 'Location', 'northwest');
title('Accuracy for each true class');

%% Figure 3: first test image of every class and what the CNN said
figure('Name', 'Example classifications', 'Position', [100 100 1000 460]);
for c = 1:numClasses
    inds = find(trueclass == c);        % as in the handout's sample code
    n = inds(1);
    subplot(2, 5, c);
    image(imageset(:,:,:,n));
    axis image off;
    if predclass(n) == c
        titleColor = [0 0.5 0];         % green = correct
    else
        titleColor = [0.8 0 0];         % red = wrong
    end
    title({['true: ' classlabels{c}], ...
           sprintf('CNN: %s (%.2f)', classlabels{predclass(n)}, classprobs(n, predclass(n)))}, ...
          'Color', titleColor);
end

%% Save for demo4_topk.m and for the video
save('cnn_results.mat', 'confusion', 'accuracy', 'predclass', 'classprobs');
fprintf('\nSaved confusion, accuracy, predclass and classprobs to cnn_results.mat\n');
