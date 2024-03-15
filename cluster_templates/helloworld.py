import sys
import os
directory = "/nethome/wplacroix/NN-SoftP/cluster_templates/hello_world/"
os.mkdir(directory)
experiment_name = "helloworld"
sys.stdout = open(f'{directory}{experiment_name}.log', 'w')
print("hello world2!")
sys.stdout.close()