import sys
temp = sys.stdout
sys.stdout = open('/data/users/wplacroix/logs/helloworld.txt', 'w')
print("hello world!")
sys.stdout.close()