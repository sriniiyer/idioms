import sys
import json

for line in sys.stdin:
  (nl, code) =  line.strip().split('\t')
  print(json.dumps({'nl': nl, 'code': code}))
