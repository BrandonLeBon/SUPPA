import argparse

def read_arguments():
    parser = argparse.ArgumentParser()
    
    #every testing
    parser.add_argument("--operator", dest="operator", required=True)
    parser.add_argument("--solver", dest="solver", required=True)
    parser.add_argument("--input_testing_dataset", dest="input_testing_dataset", required=True)
    parser.add_argument("--groundtruth_testing_dataset", dest="groundtruth_testing_dataset", required=True)
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--output_folder", dest="output_folder")
    parser.add_argument("--nb_channels", dest="nb_channels", default=3)
    parser.add_argument("--ycbcr", dest="ycbcr", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--penalty", dest="penalty", default=0.1)
    parser.add_argument("--sigma", dest="sigma", default=0.1)
    
    #unrolled only
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

    args = parser.parse_args()
    
    return args