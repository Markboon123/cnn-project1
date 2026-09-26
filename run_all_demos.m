%% RUN_ALL_DEMOS  CMPEN/EE 454 Project 1: run every demo in order.
%   Put CNNparameters.mat, cifar10testdata.mat and debuggingTest.mat in
%   this folder first. demo3_evaluate takes a few minutes.

close all;
clc;
demo1_verify_layers        % hand examples + layer-by-layer check vs debuggingTest.mat
demo2_show_layers          % intermediate feature maps and class probabilities
demo3_evaluate             % confusion matrix and accuracy on 10000 test images
demo4_topk                 % extra credit 1: top-k accuracy curve
demo5_normalize_as_conv    % extra credit 2: imnormalize as a convolution layer
