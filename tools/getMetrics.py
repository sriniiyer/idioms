from SqlMetric import SqlMetric
import argparse

def prettyPrint(score):
  (correct, total, percent) = score
  print(percent)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='getMetrics.py')

  parser.add_argument('-gold', type=str)
  parser.add_argument('-pred', type=str)
  parser.add_argument('-nl', type=str)
  parser.add_argument('-db', type=str)
  parser.add_argument('-host', type=str)
  parser.add_argument('-user', type=str)
  parser.add_argument('-passwd', type=str)
  opt = parser.parse_args()

  m = SqlMetric(opt.db, opt.pred + '.html', 'ignore')
  score = m.computeFromFiles(opt.gold, opt.pred, opt.nl, opt.host, opt.user, opt.passwd)
  prettyPrint(score)
