
class TimeoutError(Exception):
  def __init__(self):
    Exception.__init__(self,"Well, it timed out") 

