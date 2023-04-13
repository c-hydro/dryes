# Base functions for computing the Low Flow Index, using SENAMHI discharge DB 
# Developed at Fondazione CIMA, 2020
# for Python 3.7

# Used by: 
# LFI_base_stat.py and 
# LFI_Index.py

# Load required Libraries
import numpy
from datetime import datetime
from scipy.interpolate import InterpolatedUnivariateSpline
import csv

# Get list of unique codes (stations)
def get_unique_list(initList):
    uniqueList = []
    for x in initList:
        #print(x)
        if x not in uniqueList:
            uniqueList.append(x)
    return uniqueList

def count_finite_values(myList):
    count = 0
    for value in myList:
        if value is not None:
            count += 1
    return count

# Define sq object class, containing station information and statistics
class sqClass:
    def __init__(self, code, x, y, z, name, river):
        self.x = x
        self.y = y
        self.z = z
        self.code = code
        self.name = name
        self.river = river
        self.Time = []
        self.Data = []

        self.dens = None
        self.start = None
        self.end = None
        self.TimeData = None
        self.DC = []
        self.lambd_ev = []
        self.tStartev = []
        self.tEndev = []
        self.dVev = []
        self.Q_Rep = []
        self.Q95_Rep = []
        self.dV_Rep = []
        self.LFI_Rep = []
	
	# Set time
    def set_data(self, data, time):
        # Order data and time vectors
        for date in time:
            self.Time.append(datetime.strptime((date[0]), '%Y-%m-%d %H:%M:%S').date())

        for samples in data:
            self.Data.append(samples[0])

		# Sort time
        self.TimeData=sorted(zip(self.Time, self.Data), key=lambda pair: pair[0])

        #set self.start and self.end
        self.start = self.TimeData[0][0].year
        self.end = self.TimeData[-1][0].year

        #calcolates self.dens (density)
        self.dens = count_finite_values(self.Data)/(datetime.toordinal(self.TimeData[-1][0])-datetime.toordinal(self.TimeData[0][0]))
	
	# Check for enough data
    def has_enough_data(self, mstart, mend, mdens):

        if self.start <= mstart and self.end >= mend and self.dens >= mdens:
            print(str(self.name)+" has enough data")
            return True
        else:
            return False
	
	# Compute monthly discharges
    def find_monthly_DC(self, month):

        #Slices Data Vector
        slicedData=[]
        for date, sample in self.TimeData:
            if date.month == month:
                if sample is not None:
                    slicedData.append(sample)
        #PCT=range(1,99)
        PCT = [5] # percentile 5

        monthlyDC = []

        for percentile in PCT:
            monthlyDC.append(numpy.percentile(slicedData, percentile))
        return monthlyDC

    def set_DC(self):

        #For every month  we set a Vector DC length 102
        for month in range(1, 13):
            self.DC.append(self.find_monthly_DC(month))

	# Open Q95 table
    def inherit_DC(self, q95Table):
        with open(q95Table, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            skip1 = False   # !!!!!
            for row in csv_reader:
                if skip1:
                    skip1 = False
                    continue
                else:
                    if self.code == row["code"]:
                        for month in range(1, 13):
                            DCm = []
                            DCm.append(row["Q95_" + ("%02d" % month)])
                            self.DC.append(DCm) # for coherence in case of multiple percentiles
                        self.lambd_ev = row["lambda"] # get lambda value
	
	# Find low flow events for station
    def find_events(self, tsRep, teRep):

		# time and discharges
        TT, QQ = zip(*list(self.TimeData))
        # print(TT)
        QQ = list(QQ)

        # set Q95 on daily basis
        tm = []
        t95 = []
        t95_o = []
        for tt in TT:
            t15 = datetime(tt.year, tt.month, 15)
            tm.append(t15)
            t95.append(tt)
            t95_o.append(tt.toordinal())
        tm = get_unique_list(tm)
        tm_o = []
        Q95 = []
        for tt in tm:
            tm_o.append(tt.toordinal())
            month = tt.month
            Q95.append(self.DC[month - 1][0])
        spl = InterpolatedUnivariateSpline(tm_o, Q95)
        Q95 = spl(t95_o) # daily Q95

        # store for report
        iRep = 0
        for tt, Q1, Q2 in zip(TT, QQ, Q95):
            if tt.toordinal() >= tsRep and tt.toordinal() <= teRep:
                if Q1 is not None:
                    self.Q_Rep.append(Q1)
                else:
                    self.Q_Rep.append(-9998)
                self.Q95_Rep.append(Q2)
                self.dV_Rep.append(0)
                self.LFI_Rep.append(-9998)
            if tt.toordinal() < tsRep:
                iRep += 1


        # find start and end of drought spells
        kd = []
        ku = []
        for i in range(0, len(QQ)-1):
            if QQ[i+1] is not None:
                if QQ[i+1] < Q95[i+1]:
                    if QQ[i] is None:
                        kd.append(i + 1)
                    elif QQ[i] >= Q95[i]:
                        kd.append(i + 1)
            if QQ[i] is not None:
                if QQ[i] < Q95[i]:
                    if QQ[i+1] is None:
                        ku.append(i)
                    elif QQ[i+1] >= Q95[i+1]:
                        ku.append(i)
        if len(ku) > 0 and len(kd) > 0:
            if ku[0] < kd[0]:
                ku = ku[1:]
            if kd[-1] > ku[-1]:
                kd = kd[:-1]

            # adjust for minimum distance among events (10 days)
            kuu = []
            kdd = [kd[0]]
            for i in range(1, len(ku)-1):
                if (kd[i] - ku[i-1]) >= 10:
                    kuu.append(ku[i-1])
                    kdd.append(kd[i])
            kuu.append(ku[-1])

            # eliminate short events (5 days)
            kup = []
            kdo = []
            TT = list(TT)
            for i in range(0, len(kuu)):
                if (kuu[i] - kdd[i]) > 5:
                    kup.append(kuu[i])
                    kdo.append(kdd[i])
                    self.tStartev.append(TT[kdd[i]])
                    self.tEndev.append(TT[kuu[i]])

            # compute magnitude

            for ku, kd in zip(kup, kdo):
                dV = 0
                for i in range(kd, ku+1):
                    if (QQ[i] is not None) and QQ[i] < Q95[i]:
                        dV += 24*3600*(Q95[i] - QQ[i])
                for i in range(kd, ku + 1):
                    if TT[i].toordinal() >= tsRep and TT[i].toordinal() <= teRep:
                        self.dV_Rep[i - iRep] = dV
                self.dVev.append(dV)

            # store results
            self.lambd_ev = 1 / numpy.mean(self.dVev)
        else:
            self.lambd_ev = -9999

        # store LFI
        for i in range(0, len(self.dV_Rep)):
            if self.dV_Rep[i] > 0:
                self.LFI_Rep[i] = 1 - numpy.exp(- self.lambd_ev * self.dV_Rep[i])



