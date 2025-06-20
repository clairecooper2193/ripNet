# ripNet_CNN
Custom convolutional neural network for detecting hippocampal sharp wave ripples. Described in Cooper _et al._ 2025 [link](https://elifesciences.org/reviewed-preprints/101105)

The model takes as input 8 LFP channels, 4 from the cortex and 4 from the hippocampus. As the model is trained on silicon probe data in which the probe traverses the hippocampus dorso-ventrally, it therefore expects the four hippocampal channels to contain the characteristic laminar pattern ripples take going from the stratum oriens to the stratum radiatum.  

![](/images/Methods.png)
