#!/usr/bin/python

import threading
import time
import numpy as np
from threading import Thread,Semaphore

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print "Starting " + self.name
        # Get lock to synchronize threads
        #threadLock.acquire()
        print_time(self.threadID-1, self.name, self.counter, 3)
	global abc	
	abc=abc+1
	print self.name+'+++++'+str(abc)
	b.wait()
	b.wait2()
	print self.name+'-----'+str(abc)
	b.wait()
	b.wait2()
	print self.name+'-----'+str(abc)
        # Free lock to release next thread
        #threadLock.release()

class Barrier:
    def __init__(self, n):
        self.n = n
        self.count = 0
        self.mutex = Semaphore(1)
        self.turnstile = Semaphore(0)
	self.turnstile2 = Semaphore(1)

    def wait(self):
        self.mutex.acquire()
	self.count=self.count+1
	if (self.count==self.n):
		self.turnstile2.acquire()
		self.turnstile.release()
	self.mutex.release()
	self.turnstile.acquire()
	self.turnstile.release()

    def wait2(self):
        self.mutex.acquire()
	self.count=self.count-1
	if (self.count==0):
		self.turnstile.acquire()
		self.turnstile2.release()
	self.mutex.release()
	self.turnstile2.acquire()
	self.turnstile2.release()
		

#    def wait(self):
#        self.mutex.acquire()
#        self.count = self.count + 1
#	#global abc
#	#abc=8
#	print 'BARRIER COUNT '+str(self.count)
#        self.mutex.release()
#        if self.count == self.n: self.barrier.release()
#        self.barrier.acquire()
#        self.barrier.release()
#	if self.count == self.n: 
#		self.count=0
#		print '~~~~~~~~~~~~~'

def print_time(i, threadName, delay, counter):
    while counter:
        #time.sleep(delay)
	xx=x[i*5:(i+1)*5]+y[i*5:(i+1)*5]
        print "%s: %s %s" % (threadName, time.ctime(time.time()), min(xx))
        counter -= 1

#threadLock = threading.Lock()
threads = []
x=np.asarray([1,2,1,0,1,2,2,3,4,5,4,3,2,1,0])
y=np.asarray([2,5,4,3,2,1,0,6,7,8,9,8,3,1,3])
z=np.asarray([0]*3)
abc=0;
# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 1)
thread3 = myThread(3, "Thread-3", 1)

b = Barrier(4)
# Start new Threads
thread1.start()
thread2.start()
thread3.start()
print 'all reached '+str(abc)
b.wait()
abc=8
b.wait2()
print 'all reached '+str(abc)

b.wait()
abc=12
b.wait2()
print 'all reached '+str(abc)
# Add threads to thread list
threads.append(thread1)
threads.append(thread2)
threads.append(thread3)
# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"
