# CMPEN/EE 454 Project 1: CNN forward pass in MATLAB

Team: __________________________

## How to run

Put `CNNparameters.mat`, `cifar10testdata.mat` and `debuggingTest.mat` in this folder and run `run_all_demos` in MATLAB, or run any demo script on its own. The only toolbox function used is `imfilter` (Image Processing Toolbox). `demo3_evaluate` classifies all 10000 test images and takes a few minutes; it saves `cnn_results.mat`, which `demo4_topk` reads.

| File | Purpose |
|---|---|
| `apply_imnormalize.m`, `apply_relu.m`, `apply_maxpool.m`, `apply_convolve.m`, `apply_fullconnect.m`, `apply_softmax.m` | the six layer functions, with the names and signatures from the handout |
| `run_layer.m` | applies one layer, chosen by its name in `layertypes` |
| `cnn_forward.m` | runs the 18 layers in order on one image and returns every layer's output |
| `demo1_verify_layers.m` | hand-checkable examples, then every layer compared with `debuggingTest.mat` |
| `demo2_show_layers.m` | feature maps of every layer and the score/probability bar charts for one image |
| `demo3_evaluate.m` | confusion matrix, classification rate and per-class rates on the 10000 test images |
| `demo4_topk.m` | extra credit 1: top-k accuracy curve |
| `demo5_normalize_as_conv.m` | extra credit 2: layer 1 done by an ordinary convolution layer |
| `show_all_layers.m`, `show_channels.m` | display helpers used by demo2 |

## Implementation notes

**Normalization** converts the uint8 image to double before any arithmetic. In uint8, results are rounded and clipped to 0..255, so subtracting the mean would turn every negative value into 0.

**Convolution** follows the handout's formula directly: for each output channel l, the D1 results of `imfilter(In(:,:,k), F(:,:,k,l), 'conv', 'same', 0)` are added up and the bias b_l is added. `'conv'` rotates the filter 180 degrees, `'same'` keeps the N x M size, and `0` zero-pads. Using correlation instead is equivalent to convolving with rotated filters; demo1 (Part 4) shows that this makes layer 2 of the debugging image wrong by up to 2.76 and changes the network's answer from airplane to deer.

**Maxpool** uses the method of Figure 1B: four subsampled images that start at (1,1), (1,2), (2,1) and (2,2), combined two at a time with `max`. Using `:` for the third index pools all channels in one step.

**Fully connected**: each output is `sum(Fl(:) .* In(:)) + b_l`, i.e. the triple sum from the handout written as a dot product of the unrolled filter and input. `(:)` unrolls both arrays in the same column-major order, so matching (i,j,k) entries are multiplied.

**Softmax** subtracts alpha = max score before `exp`, as the handout specifies. It gives the same answer, but prevents overflow: for scores [1000 1001 1002] the version without alpha returns NaN.

**Main routine**: `cnn_forward` loops over d = 1..18 and `run_layer` picks the function from `layertypes{d}`, so the layer order comes straight from `CNNparameters.mat` (Table 1). The predicted class is the one with the highest probability; if two were exactly tied, `max` would return the lower class index, but no ties occur on the test set.

## Results

Every layer agrees with `debuggingTest.mat` to about 1e-15, both when each layer is run on its own and when the whole network is run end to end (demo1 uses a pass tolerance of 1e-10). The debugging image is classified as airplane with probability 0.8469.

On the 10000 test images the classification rate is **61.56%** (6156 correct). Confusion matrix, rows = true class, columns = class output by the CNN:

| | airpl | autom | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | **665** | 28 | 74 | 28 | 10 | 31 | 8 | 25 | 93 | 38 |
| **automobile** | 49 | **725** | 4 | 10 | 2 | 20 | 14 | 16 | 29 | 131 |
| **bird** | 80 | 7 | **411** | 72 | 98 | 188 | 98 | 29 | 11 | 6 |
| **cat** | 18 | 11 | 54 | **322** | 30 | 431 | 77 | 30 | 13 | 14 |
| **deer** | 25 | 6 | 56 | 60 | **469** | 131 | 111 | 130 | 11 | 1 |
| **dog** | 10 | 4 | 42 | 119 | 36 | **722** | 19 | 41 | 2 | 5 |
| **frog** | 11 | 2 | 36 | 71 | 42 | 72 | **747** | 14 | 1 | 4 |
| **horse** | 22 | 1 | 26 | 37 | 56 | 165 | 11 | **661** | 3 | 18 |
| **ship** | 135 | 35 | 19 | 26 | 5 | 25 | 11 | 6 | **706** | 32 |
| **truck** | 49 | 100 | 6 | 21 | 1 | 26 | 10 | 31 | 28 | **728** |

Per-class rates range from 74.7% (frog) and 72.8% (truck) down to 46.9% (deer), 41.1% (bird) and 32.2% (cat). The largest confusion is cat classified as dog (431 times), followed by bird as dog (188), horse as dog (165), ship as airplane (135), deer as dog (131) and automobile as truck (131). The network outputs "dog" 1811 times although only 1000 test images are dogs, so dog's high rate comes partly from dog being its default answer for animals (only 39.9% of its "dog" answers are right). Most mistakes stay within a group: 81.4% of the 3844 errors confuse one animal with another or one vehicle with another.

Top-k accuracy (extra credit 1), k = 1..10: 61.56, 80.31, 88.84, 93.45, 96.25, 97.93, 98.93, 99.45, 99.83, 100 percent. With two guesses the correct class is included 80% of the time.

## Extra credit 2: imnormalize as a convolution layer

For each channel l, normalization computes

    Out(:,:,l) = (In(:,:,l)/255 - mu_l) / sigma_l  =  (1/(255 sigma_l)) In(:,:,l)  -  mu_l/sigma_l

which is a scale plus an offset per channel. A convolution layer computes

    Out(:,:,l) = sum_k F_l(:,:,k) * In(:,:,k) + b_l

so the two are identical if we use a **1 x 1 x 3 x 3 filterbank** with F_l(1,1,k) = 1/(255 sigma_l) when k = l and 0 otherwise, and a **bias vector of length 3** with b_l = -mu_l/sigma_l. With the CIFAR-10 values the three nonzero weights are 1/(255 x 0.2470) = 0.015877, 1/(255 x 0.2435) = 0.016105 and 1/(255 x 0.2616) = 0.014991, and the biases are -1.98947, -1.98029 and -1.70680.

A 1 x 1 filter reads no neighbouring pixels, so the 180-degree flip and the zero padding never come into play, and the zero weights for k different from l keep the color channels separate. The same result comes from 3 x 3 filters that are zero except for the center tap, which would match the size of the other convolution layers. The only requirement is that the uint8 image is converted to double first (`apply_convolve` does this). `demo5_normalize_as_conv` builds this filterbank, shows it reproduces `apply_imnormalize` to about 1e-15, and runs the whole network with layer 1 replaced by it.

Why not fold normalization into the layer-2 filters, since both are linear? It is exact in the interior, but not at the border: layer 2 pads the *normalized* image with 0, which corresponds to a raw value of 255 mu_l, while the folded version would pad the raw image with 0. Keeping normalization as its own 1 x 1 layer avoids that mismatch.
