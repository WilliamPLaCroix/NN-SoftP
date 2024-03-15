import sys
sys.stdout = open('/nethome/wplacroix/NN-SoftP/cluster_templates/helloworld.log', 'w')
print("hello world2!")
sys.stdout.close()