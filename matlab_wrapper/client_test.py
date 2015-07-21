__author__ = 'jeremy'
# to run locally:
# ssh -i key_file.pem -L 18861:localhost:18861 ubuntu@extremeli.trendi.guru)
import sys

from matlab_wrapper import matlab_client

sys.path.append("~/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules")

mateng = matlab_client.Engine()

# Now you can call Matlab functions from python
mateng.factor(1011)
# >> matlab.int64([[3,337]])
mateng.isprime(1013)
# >> True