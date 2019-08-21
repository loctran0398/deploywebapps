# -*- encoding: utf-8 -*-
#
# Copyright (c) 2016 Chronos AS
#
# Authors: Fredrik Stormo, Stefan Remman
# Contact: kjetil.karlsen@chronosit.no

from functools import wraps
import os,errno
import signal as sig

class TimeoutException(Exception):
  pass

def timeout(seconds=5, error_message=os.strerror(errno.ETIME)):
  def decorator(function):
    def has_timed_out(sig, frame):
      raise TimeoutException(error_message)

    def wrapper(*args, **kwargs):
      sig.signal(sig.SIGALRM, has_timed_out)
      sig.alarm(seconds)
      try:
        result = function(*args, **kwargs)
      finally:
        sig.alarm(0)
      return result
    return wraps(function)(wrapper)
  return decorator

