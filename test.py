import sys
import json

jsonString = json.dumps({ 'text1': sys.argv[1], 'text2': sys.argv[2] })
print(jsonString)