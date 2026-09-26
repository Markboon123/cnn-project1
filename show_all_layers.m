function show_all_layers(layerResults, layertypes)
%SHOW_ALL_LAYERS  One figure with every channel of every image-shaped layer.
%   show_all_layers(layerResults, layertypes)
%   Row d shows the channels of layer d as greyscale tiles. Each tile is
%   contrast-stretched to its own min..max, and smaller layers are enlarged
%   by pixel replication (kron) so every tile is 32 x 32. The blockiness of
%   the pooled layers is therefore their true resolution.

tile = 32;      % displayed size of every tile
gap  = 2;       % white border between tiles

% only the layers that are still images (layers 17 and 18 are 1 x 1 x 10)
layers = [];
for d = 1:length(layerResults)
    if size(layerResults{d}, 1) > 1
        layers(end+1) = d; %#ok<AGROW>
    end
end
nrows = numel(layers);
ncols = 0;
for d = layers
    ncols = max(ncols, size(layerResults{d}, 3));
end

mosaic = ones(gap + nrows*(tile+gap), gap + ncols*(tile+gap));   % white background
for r = 1:nrows
    arr   = layerResults{layers(r)};
    scale = tile / size(arr, 1);                 % 1, 2, 4 or 8
    for k = 1:size(arr, 3)
        t  = arr(:,:,k);
        lo = min(t(:));
        hi = max(t(:));
        if hi > lo
            t = (t - lo) / (hi - lo);            % stretch to 0..1
        else
            t = zeros(size(t));                  % constant channel, e.g. all 0 after ReLU
        end
        t = kron(t, ones(scale));                % each pixel becomes a scale x scale block
        rows = gap + (r-1)*(tile+gap) + (1:tile);
        cols = gap + (k-1)*(tile+gap) + (1:tile);
        mosaic(rows, cols) = t;
    end
end

% one label per row: layer number, type and output size
rowLabels = cell(nrows, 1);
for r = 1:nrows
    d  = layers(r);
    sz = size(layerResults{d});
    rowLabels{r} = sprintf('%d  %s  %dx%dx%d', d, layertypes{d}, sz(1), sz(2), sz(3));
end

figure('Name', 'All layers', 'Position', [100 50 800 950]);
imagesc(mosaic, [0 1]);
colormap(gray);
axis image;
set(gca, 'YTick', gap + ((1:nrows) - 1)*(tile+gap) + tile/2, 'YTickLabel', rowLabels, ...
         'XTick', gap + ((1:ncols) - 1)*(tile+gap) + tile/2, 'XTickLabel', num2str((1:ncols)'), ...
         'XAxisLocation', 'top', 'TickLength', [0 0]);
xlabel('channel');
end
