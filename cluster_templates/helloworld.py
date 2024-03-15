import sys
temp = sys.stdout
sys.stdout = open('/nethome/wplacroix/NN-SoftP/cluster_templates/helloworld.log', 'w')
sys.stdout.write("hello world!")
sys.stdout.close()
sys.stdout = temp