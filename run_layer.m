function outarray = run_layer(inarray, layertype, filterbank, biasvals)
%RUN_LAYER  Apply one CNN layer, chosen by its type name.
%   outarray = run_layer(inarray, layertype, filterbank, biasvals)
%   layertype is one of the strings stored in layertypes (CNNparameters.mat):
%   'imnormalize', 'convolve', 'relu', 'maxpool', 'fullconnect', 'softmax'.
%   filterbank and biasvals are only used by 'convolve' and 'fullconnect'
%   (for the other layers their cells in CNNparameters.mat are empty).

switch layertype
    case 'imnormalize'
        outarray = apply_imnormalize(inarray);
    case 'convolve'
        outarray = apply_convolve(inarray, filterbank, biasvals);
    case 'relu'
        outarray = apply_relu(inarray);
    case 'maxpool'
        outarray = apply_maxpool(inarray);
    case 'fullconnect'
        outarray = apply_fullconnect(inarray, filterbank, biasvals);
    case 'softmax'
        outarray = apply_softmax(inarray);
    otherwise
        error('run_layer: unknown layer type ''%s''', layertype);
end
end
