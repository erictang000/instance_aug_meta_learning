## InstaAug
* Initialize some learnable invariance module using InstaAug_module code
* We need to apply the instance augmentation before we forward pass on the model
    * look at the code in the predict function for model_wrapper.py for how this works? (lines 297-410)
        * could also see if there's easier implementation in self-supervised/train.py for this part
    * hopefully this is mostly just forward passing through the instance aug module and dealing with configs
* after we calculate and backprop the loss for the model, we should do the same for the instaAug module
    * Depending what optimizer is being used, this can be using same optimizer or different 
    * set model to eval -> optimizer zero grad -> backprop instaaug module -> step instaaug module -> set model back to train