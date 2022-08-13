import argparse

def read_arguments():
    parser = argparse.ArgumentParser()
    
    #every training
    parser.add_argument("--operator", dest="operator", required=True)
    parser.add_argument("--solver", dest="solver", required=True)
    parser.add_argument("--input_training_dataset", dest="input_training_dataset", required=True)
    parser.add_argument("--groundtruth_training_dataset", dest="groundtruth_training_dataset", required=True)
    parser.add_argument("--input_testing_dataset", dest="input_testing_dataset", required=True)
    parser.add_argument("--groundtruth_testing_dataset", dest="groundtruth_testing_dataset", required=True)
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--epochs", dest="epochs", default=75)
    parser.add_argument("--batch_size", dest="batch_size", default=16)
    parser.add_argument("--learning_rate", dest="learning_rate", default=0.0001)
    parser.add_argument("--nb_channels", dest="nb_channels", default=3)
    parser.add_argument("--nb_samples", dest="nb_samples", default=1)
    parser.add_argument("--patch_size", dest="patch_size", default=0)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--penalty", dest="penalty", default=0.1)
    parser.add_argument("--sigma", dest="sigma", default=0.1)
    
    #unrolled based solvers only
    parser.add_argument("--recurrent", dest="recurrent", action="store_true")
    parser.add_argument("--nb_block", dest="nb_block", default=6)
    
    #implicit solvers only
    parser.add_argument("--max_iter", dest="max_iter", default=50)
    parser.add_argument("--tol", dest="tol", default=1e-3)
    parser.add_argument("--fp_algorithm", dest="fp_algorithm", default="anderson")
    
    #gaussian blurring only
    parser.add_argument("--kernel_sigma", dest="kernel_sigma", default=2.0)
    
    #downsampling only
    parser.add_argument("--scale", dest="scale", default=1.0)
    parser.add_argument("--kernel_type", dest="kernel_type", default='bicubic')
    parser.add_argument("--antialiasing", dest="antialiasing", action="store_true")
    
    #SIUPPA solver only
    parser.add_argument("--nb_samples_per_inference", dest="nb_samples_per_inference", default=1)
    parser.add_argument("--lambda_lr", dest="lambda_lr", default=0.1)

    args = parser.parse_args()
    
    return args