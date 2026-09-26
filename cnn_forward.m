function layerResults = cnn_forward(imrgb, layertypes, filterbanks, biasvectors)
%CNN_FORWARD  Forward pass of the 18-layer CNN (Table 1) on one image.
%   layerResults = cnn_forward(imrgb, layertypes, filterbanks, biasvectors)
%   imrgb        : 32 x 32 x 3 uint8 color image
%   layertypes, filterbanks, biasvectors : cell arrays from CNNparameters.mat
%   layerResults : 1 x 18 cell array; layerResults{d} is the output of layer d
%                  (same layout as layerResults in debuggingTest.mat), and
%                  layerResults{end} is the 1 x 1 x 10 vector of class probabilities
%
%   Every layer takes the previous layer's output as its input. All
%   intermediate outputs are kept so the demos can display them.

numLayers    = length(layertypes);
layerResults = cell(1, numLayers);

x = imrgb;                                   % input to layer 1
for d = 1:numLayers
    x = run_layer(x, layertypes{d}, filterbanks{d}, biasvectors{d});
    layerResults{d} = x;                     % becomes the input to layer d+1
end
end
