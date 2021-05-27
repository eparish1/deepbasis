import numpy as np
import os
import sys

def main(argv):
    with open('ood_pp_dropout_deep_basis_depth_'+str(argv[0])+'_basis_'+str(argv[1])+'.batch', 'w+') as the_file:
        the_file.write('#!/bin/bash\n')
        the_file.write('\n')
        the_file.write('#SBATCH -o ood_pp_dropout_drop_deep_basis_depth_'+str(argv[0])+'_basis_'+str(argv[1])+'.out\n')
        the_file.write('#SBATCH -p zen\n')
        the_file.write('#SBATCH -t 10:00:00\n')
        the_file.write('\n')
        the_file.write('python3 yk_postProcessModels.py  '+str(argv[0])+' '+str(argv[1])+'\n')
        #the_file.write('python3 driver_yuki_v2.py '+str(argv[0])+' '+str(argv[1])+'\n')
    os.system('sbatch ood_pp_dropout_deep_basis_depth_'+str(argv[0])+'_basis_'+str(argv[1])+'.batch')
if __name__ == '__main__':
    main(sys.argv[1:])
