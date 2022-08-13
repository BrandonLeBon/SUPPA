import argparse

def default_read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", dest="input_dataset", required=True)
    parser.add_argument("--output_dataset", dest="output_dataset", required=True)
    parser.add_argument("--nb_channels", dest="nb_channels", default=3)
    parser.add_argument("--noise_sigma", dest="noise_sigma", default=0.0)
    
    return parser

def gaussian_blur_read_arguments():
    parser = default_read_arguments()
    parser.add_argument("--kernel_sigma", dest="kernel_sigma", required=True)
    
    args = parser.parse_args()
    
    return args.input_dataset, args.output_dataset, args.nb_channels, float(args.kernel_sigma), float(args.noise_sigma)

def mosaicing_read_arguments():
    parser = default_read_arguments()
    
    args = parser.parse_args()
    
    return args.input_dataset, args.output_dataset, args.nb_channels, float(args.noise_sigma)

def downsampling_read_arguments():
    parser = default_read_arguments()

    parser.add_argument("--scale", dest="scale", required=True)
    parser.add_argument("--kernel_type", dest="kernel_type", default='bicubic')
    parser.add_argument("--antialiasing", dest="antialiasing", action="store_true")
    
    args = parser.parse_args()
    
    return args.input_dataset, args.output_dataset, args.nb_channels, float(args.noise_sigma), int(args.scale), args.kernel_type, args.antialiasing