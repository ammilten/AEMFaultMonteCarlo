import MonteCarlo
import sys, getopt, os

# ---------------- Parse arguments ---------------------
def get_args(argv):
    Q = 14
    N = 1
    showDists = False
    parallel = False
    nproc = None
    MCfolder = ''
    foundMC = False
    overwrite = False
    imp = False

    try:
        opts, args = getopt.getopt(argv,"hf:s:n:p:doi",["mcfolder=","spacing=","nreals=","showdists","parallel","overwrite","import"])
    except getopt.GetoptError:
        print("ERTMonteCarlo.py -f <path_to_mcfolder> -s <max_spacing> -n <num_reals> -p <num_processors> -d -o")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("\nBASIC USAGE:\npython ERTFaultMonteCarlo.py -f <path_to_mcfolder> -n <num_reals>\n\nSWITCHES:\n  -f, --mcfolder <folder>: Path to Monte Carlo output folder (required)\n  -s, --spacing <max_spacing>: Max grid spacing. (default=25)\n  -n, --nreals <num_reals>: Number of realizations (default=1)\n  -p, --parallel <num_processors>: If specified, run in parallel. Option to specify number of of processors. If number of processors is not specified, it is auto-selected.\n  -d, --showdists: If specified, plot the parameter distributions prior to simulations.\n  -o, --override: If specified, the Monte Carlo will overwrite the data in MCfolder if MCfolder already exists.\n  -i, --import: If specified, the program will attempt to load Monte Carlo parameters from <MCfolder>/params.dat.")
            sys.exit(2)
        elif opt in ("-s", "--spacing"):
            dh = float(arg)
        elif opt in ("-n", "--nreals"): 
            N = int(arg)
        elif opt in ("-f", "--mcfolder"):
            MCfolder = arg
            foundMC = True
        elif opt in ("-d", "--showdists"):
            showDists = True
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("-i", "--import"):
            imp = True
        elif opt in ("-p", "--parallel"):
            parallel = True
            try:
                nproc = int(arg)
            except:
                nproc = None
             
    if not foundMC:
        print("MCfolder argument required: use '-f <path_to_mcfolder>' or '--mcfolder <path_to_mcfolder>'")
        sys.exit(2)

    if not MCfolder.endswith('/'):
        MCfolder = MCfolder+'/'

    print("MCfolder = "+MCfolder)
    print("Max Spacing = " + str(dh) + "meters")
    print("Number of realizations = " + str(N))
    if parallel:
        if nproc is None:
            print("Parallelized = true (auto-select number of processors)")
        else:
            print("Parallelized = true (" + str(nproc) + " processors)")
    else:
        print("Parallelized = false")
    if showDists:
        print("Plotting distributions = true")
    else:
        print("Plotting distributions = false")
    if imp:
        print("\n** Simulation will be imported from '" + MCfolder + "params.dat'.\n   Mesh quality and number of realizations above may be incorrect.")

    print("")
    return MCfolder, dh, N, showDists, parallel, nproc, overwrite, imp



#------------------ MAIN SCRIPT -----------------------
if __name__ == "__main__":

    print("\n----------- Preparing Monte Carlo -----------")
    MCfolder, dh, N, showDists, parallel, nproc, overwrite, imp = get_args(sys.argv[1:])

#    MCfolder = '/home/ammilten/Programs/ERTFaultMonteCarlo/data/MC/'

    dip = [25, 165] #uniform
    H = [120, 40] #normal
    xpos = [400,40] #normal
    rho_fault = [20, 10, .45] #lognormal
    rho_back = [40, 30, .8] #lognormal

    dep = 1000
    xtra = 1000
    ny = 11
    y0 = 100

    if imp:
        MC = MonteCarlo.import_simulation(MCfolder,overwrite, parallel, nproc, showDists)
    else:
        MC = MonteCarlo.MonteCarlo(dep, xtra, dh, y0, ny, dip, H, xpos, rho_fault, rho_back, N, MCfolder, overwrite, parallel, nproc, showDists)

    MC.run()

