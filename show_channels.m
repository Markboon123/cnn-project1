function show_channels(arr, name)
%SHOW_CHANNELS  Display every channel of an N x M x D array as a greyscale image.
%   show_channels(arr, name)
%   Each channel is contrast-stretched on its own (imagesc), so the grey
%   levels show the pattern inside a channel. Signed outputs (convolution)
%   put zero at some mid grey; after ReLU, zero is black.

D = size(arr, 3);
ncols = min(D, 5);
nrows = ceil(D / ncols);

figure('Name', name, 'Position', [100 100 1000 460]);
for k = 1:D
    subplot(nrows, ncols, k);
    imagesc(arr(:,:,k));
    axis image off;
    title(sprintf('channel %d', k));
end
colormap(gray);
sgtitle(sprintf('%s: %d x %d x %d', name, size(arr,1), size(arr,2), D));
end
