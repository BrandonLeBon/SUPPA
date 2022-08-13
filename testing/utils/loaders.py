import sys

def load_operator(args, device):
    if args.operator=="gaussian_blurring":
        from operators.blurring import GaussianBlur
        return GaussianBlur(kernel_sigma=float(args.kernel_sigma), nb_channels=int(args.nb_channels), device=device).to(device=device)
    elif args.operator=="mosaicing":
        from operators.mosaicing import Mosaicing
        return Mosaicing(device=device).to(device=device)
    elif args.operator=="downsampling":
        from operators.downsampling import Downsampling
        return Downsampling(scale=int(args.scale), kernel_type=args.kernel_type, antialiasing=args.antialiasing, device=device).to(device=device)
    else:
        print("Unknown operator")
        sys.exit()
        
def load_solver(args, operator, device):
    if args.solver=="unrolled":
        from solvers.unrolled import Unrolled
        return Unrolled(linear_operator=operator, nb_channels=int(args.nb_channels), nb_block=int(args.nb_block), recurrent=args.recurrent, pretrained=args.pretrained, penalty_initial_val=float(args.penalty), sigma_initial_val=float(args.sigma), device=device).to(device=device)
    elif args.solver=="DEQ":
        from solvers.DEQ import DEQ
        return DEQ(linear_operator=operator, nb_channels=int(args.nb_channels), pretrained=args.pretrained, solver=args.fp_algorithm, max_iter=int(args.max_iter), tol=float(args.tol), penalty_initial_val=float(args.penalty), sigma_initial_val=float(args.sigma), device=device).to(device=device)
    elif args.solver=="JFBI":
        from solvers.JFBI import JFBI
        return JFBI(linear_operator=operator, nb_channels=int(args.nb_channels), pretrained=args.pretrained, solver=args.fp_algorithm, max_iter=int(args.max_iter), tol=float(args.tol), penalty_initial_val=float(args.penalty), sigma_initial_val=float(args.sigma), device=device).to(device=device)
    elif args.solver=="SIUPPA":
        from solvers.SIUPPA import SIUPPA
        return SIUPPA(linear_operator=operator, nb_channels=int(args.nb_channels), pretrained=args.pretrained, solver=args.fp_algorithm, max_iter=int(args.max_iter), tol=float(args.tol), penalty_initial_val=float(args.penalty), sigma_initial_val=float(args.sigma), device=device).to(device=device)
    else:
        print("Unknown solver")
        sys.exit()