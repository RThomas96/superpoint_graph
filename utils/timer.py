import numpy as np
import time
# Record a set of interval
class Timer:
    def __init__(self, step):
        self.times = np.zeros(step)

    def start(self, step):
        self.times[step] = time.perf_counter()

    def stop(self, step):
        self.times[step] =  time.perf_counter() - self.times[step]

    def getFormattedTimer(self, names):
        strTime = ""
        for i, x in enumerate(self.times):
            strTime += names[i]
            strTime += ": " 
            if x > 3600:
                strTime += str(round(x / 60. / 60., 2)) 
                strTime += "h / "
            elif x > 60:
                strTime += str(round(x / 60., 2)) 
                strTime += "min / "
            else:
                strTime += str(round(x, 2)) 
                strTime += "s / "
        return strTime[:-2]

    def reset(self):
        for i in self.times:
            i = 0

