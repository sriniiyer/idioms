import MySQLdb
import warnings
import sys
from Timeout import TimeoutError
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def runQuery(q, timeout, dbconn):
  q = q.replace(' (','(').replace('< =', '<=').replace('> =', '>=').replace('< >', '<>').replace('! =', '!=')
  q = q.replace(" / ", "/").replace(" / ", "/")
  q = re.sub('SELECT ','SELECT /*+ MAX_EXECUTION_TIME(' + str(timeout) + ') */ ', q, 1)
  q = re.sub('select ','SELECT /*+ MAX_EXECUTION_TIME(' + str(timeout)+ ') */ ', q, 1)
  cu = dbconn.cursor()
  cu.execute(q)
  result = {}
  result['tuples'] = cu.fetchall()
  result['status'] = True
  result['row_count'] = cu.rowcount
  if cu.description:
    result['fields'] = [
        {'name': col[0], 'type': col[1]} for col in cu.description]
  cu.close()
  return result

def compute(dbName, num, gold, q1, nl, host, user, passwd):

  for i in range(0, 5):
    try:
      dbconn = MySQLdb.connect(
        host=host,
        user=user,
        passwd=passwd,
        db=dbName
      )
      break
    except MySQLdb._exceptions.OperationalError:
      time.sleep(5)


  warnings.filterwarnings('ignore', category=MySQLdb.Warning)
  success = 1

  # Run predicted query
  try:
    res1 = runQuery(q1, 10000, dbconn)
    res1 = res1['tuples']
  except MySQLdb._exceptions.ProgrammingError:
    res1 = []
    success = 2
  except MySQLdb._exceptions.InterfaceError:
    success = 0
    res1 = []
  except MySQLdb._exceptions.OperationalError:
    res1 = []
  except MySQLdb._exceptions.NotSupportedError:
    success = 0
    res1 = []

  # Run gold query
  try:
    res2 = runQuery(gold, 10000, dbconn)
    res2 = res2['tuples']
  except MySQLdb._exceptions.ProgrammingError:
    import ipdb
    ipdb.set_trace()
    raise ValueError('Gold Query does not run')
  except MySQLdb._exceptions.OperationalError:
    res2 = []
  except MySQLdb._exceptions.InterfaceError:
    import ipdb
    ipdb.set_trace()
    res2 = []

  dbconn.close()

  res1_set = set()
  for x in res1:
    res1_set.add(x)

  res2_set = set()
  for x in res2:
    res2_set.add(x)

  if res1_set != res2_set:
    return (0, success, 0, num, gold, q1, nl)
  return (1, success, 0, num, gold, q1, nl)


class SqlMetric:

  def __init__(self, db, debugFile, warn):
    self.dbName = db
    self.db = None
    self.debugFile = open(debugFile, 'w') if debugFile != '' else None

  def computeFromFiles(self, goldFile, predFile, nlFile, host, user, passwd):
    warnings.filterwarnings('ignore', category=MySQLdb.Warning)
    pool = ThreadPoolExecutor(10)
    score = 0
    num = 0.0
    goldFile = open(goldFile, 'r')
    predFile = open(predFile, 'r')
    nlFile = open(nlFile, 'r')
    futures = []
    for g, m, nl in zip(goldFile, predFile, nlFile):
      futures.append(pool.submit(compute, self.dbName, num, g.strip(), m.strip(), nl.strip(), host, user, passwd))
      num += 1

    #print(num)
    for x in as_completed(futures):
      (s, success, err, n, g, m, nl) = x.result()
      if self.debugFile:
        self.debugFile.write('{}<br><b>Nl:</b>{}<br><br><b>Gold:</b><br>{}<br><br><b>Pred:</b><br>{}<br><br><b>Result:</b>{}<br><b>Success:</b>{}<br><br><br>'.format(n + 1, nl, g, m, "W" if s == 0 else "R", "" if success == 1 else ("NE" if success == 2 else "F") ))
      score += s

    if self.debugFile:
      self.debugFile.close()
    return (score, num, score * 100.0 / num)

  def computeFromMaps(self, goldMap, methodMap):
    score = 0
    num = 0.0

    for key in goldMap:
      if key in methodMap:
        if self.debugFile:
          sys.stdout.write(str(key) + ' ')
        (s, success) = self.compute(goldMap[key][0], methodMap[key][0])
        if self.debugFile:
          self.debugFile.write(str(key) + '\t' + goldMap[key][0] + '\t' + methodMap[key][0] + '\t' + str("W" if s == 0 else "R" ) + '\t' + str("" if success == 1 else "F") + '\n')
        score += s
        num += 1

    if self.debugFile:
      self.debugFile.close()
    return (score, num, score * 100.0 / num)

if __name__ == '__main__':
  b = SqlMetric('atis', False, '', 'error')
  b.runQuery('SELECT sum(from_airport) from flight;')
