# Compute Low Flow Index, using SENAMHI ground stations DB 
# Developed at Fondazione CIMA, 2020
# for Python 3.7

# Configuration file needed: LFI_config.json

# Load required Libraries
import os
import csv
import jaydebeapi
import json
# import datetime
import calendar

# sq_class contains most of the tailored function to access and use SENAMHI discharge "q" DB
from sq_class import *

# Connect to SENAMHI discharge DB (MDB), using UCanAccess tool
def new_db_connection(fileName):
    dirPath = os.path.dirname(os.path.realpath(__file__))
#    fullName = dirPath + "/" + fileName
    fullName = fileName
    accessAdd = dirPath + "/UCanAccess-5.0.0-bin"
    fileList = [
        accessAdd + "/ucanaccess-5.0.0.jar",
        accessAdd + "/lib/commons-lang3-3.8.1.jar",
        accessAdd + "/lib/commons-logging-1.2.jar",
        accessAdd + "/lib/hsqldb-2.5.0.jar",
        accessAdd + "/lib/jackcess-3.0.1.jar",
    ]

    classPath = ":".join(fileList)
    temp = "jdbc:ucanaccess://" + fullName
	
	# define connection routine
    conn = jaydebeapi.connect(
        "net.ucanaccess.jdbc.UcanaccessDriver",
        temp,
        ["", ""],
        classPath
    )
    return conn

def fetch_db_data(conn, sqlStr):
    cursor = conn.cursor()
    cursor.execute(sqlStr)
    return cursor.fetchall()

# Create list of valid discharge values
def get_valid_list(initList, q95tab):
    validList = []
#    line_count = 0
    good_codes = []
    for row in q95tab:
#        if line_count > 0:
        good_codes.append(row["code"])
#        line_count += 1
    for x in initList:
        #print(x)
        if x[0] in good_codes:
            if x not in validList:
                validList.append(x[0])
    return validList

# Create table of low flow events (csv), from sq object (per station)
def create_csv_events(sq, evFolder):
    fileName = os.path.join(evFolder, sq.code + "-LowFlowEvents.csv")
    with open(fileName, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        row = []
        # header
        row.append("Start")
        row.append("End")
        row.append("Dvol")
        row.append("LFI")
        writer.writerow(row)
        for tstart, tend, dV in zip(sq.tStartev, sq.tEndev, sq.dVev):
            row = []
            row.append(tstart) # event start
            row.append(tend) # event end
            row.append(dV) # volume missing
            row.append(1 - numpy.exp(- sq.lambd_ev * dV)) # LFI value
            writer.writerow(row)
            # print(row)

# Create table of latest event (csv)
def create_csv_latest(latestTable, latestCode, latestStart, latestEnd, latestLFI):
    with open(latestTable, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        row = []
        # header
        row.append("Code")
        row.append("Start")
        row.append("End")
        row.append("LFI")
        writer.writerow(row)
        for code, tstart, tend, LFI in zip(latestCode, latestStart, latestEnd, latestLFI):
            row = []
            row.append(code) # station code
            row.append(tstart) # event start
            row.append(tend) # event end
            row.append(LFI) # LFI value
            writer.writerow(row)
            print(row)

# create data for Dewetra output (txt)
def create_reports(sq, destdLFI, destdQQQ, destdQ95, tstart, tend):
    tstartstr = "{:04d}".format(tstart.year) + "{:02d}".format(tstart.month) \
              + "{:02d}".format(tstart.day) + "0000"
    tendstr = "{:04d}".format(tend.year) + "{:02d}".format(tend.month) \
              + "{:02d}".format(tend.day) + "0000"
    # LFI value
    fileName = os.path.join(destdLFI, "hydrograph_Bolivia_" + sq.code
                            + "_" + tendstr + ".txt")
    go_write_report(fileName, tstartstr, tendstr, sq.LFI_Rep, [])
    
    # Q observed
    fileName = os.path.join(destdQQQ, "hydrograph_Bolivia_" + sq.code
                            + "_" + tendstr + ".txt")
    go_write_report(fileName, tstartstr, tendstr, sq.Q_Rep, sq.Q95_Rep)
    # Q 95 series
    fileName = os.path.join(destdQ95, "hydrograph_Bolivia_" + sq.code
                            + "_" + tendstr + ".txt")
    go_write_report(fileName, tstartstr, tendstr, sq.Q95_Rep, [])

# Create Dewetra output (txt)
def go_write_report(fileName, tstartstr, tendstr, VV, V2):
    fid = open(fileName, 'w')
    # header
    fid.write("Procedure=LFI\n")
    fid.write("DateMeteoModel=" + tendstr + "\n")
    fid.write("DateStart=" + tstartstr + "\n")
    fid.write("Temp.Resolution=1440\n")
    fid.write("SscenariosNumber=1\n")
    # insert Observed Q
    if len(VV) > 0:
        Vstr = ""
        for i in range(0, len(VV)):
            Vstr = Vstr + "{:9.3f}".format(VV[i]) + " "
        fid.write(Vstr + "\n")
    # insert Q95 series
    if len(V2) > 0:
        Vstr = ""
        for i in range(0, len(V2)):
            Vstr = Vstr + "{:9.3f}".format(V2[i]) + " "
        fid.write(Vstr + "\n")
    fid.close()


if __name__ == '__main__':
    # start main program
    print("Hello world")
	
	# Read configuration file (in same folder)
    jsoname = 'LFI_config.json'
    with open(jsoname) as jf:
        params = json.load(jf)
        dbFileName = params["DataBase"]
        q95Table = params["Q95Table"]
        indFold = params["Index_folder"]
        QQQFold = params["Qday_folder"]
        Q95Fold = params["Q95_folder"]
        startYear = params["StartYear4LFI"]
        startMonth = params["StartMonth4LFI"]
        endYear = params["EndYear4LFI"]
        endMonth = params["EndMonth4LFI"]
        startRepYear = params["StartYearReport"]
        startRepMonth = params["StartMonthReport"]
        endRepYear = params["EndYearReport"]
        endRepMonth = params["EndMonthReport"]
        mdens = params["MinDataDens"]


    # adjust output folders (create if do not exist)
    desty = os.path.join(indFold, "{:04d}".format(endRepYear))
    if not (os.path.isdir(desty)):
        os.mkdir(desty)
    destm = os.path.join(desty, "{:02d}".format(endRepMonth))
    if not (os.path.isdir(destm)):
        os.mkdir(destm)
    destd = os.path.join(destm,
                         "{:02d}".format(calendar.monthrange(endRepYear, endRepMonth)[1]))
    if not (os.path.isdir(destd)):
        os.mkdir(destd)

    evFolder = os.path.join(destd, "Events")
    if not (os.path.isdir(evFolder)):
        os.mkdir(evFolder)

    desty = os.path.join(QQQFold, "{:04d}".format(endRepYear))
    if not (os.path.isdir(desty)):
        os.mkdir(desty)
    destm = os.path.join(desty, "{:02d}".format(endRepMonth))
    if not (os.path.isdir(destm)):
        os.mkdir(destm)
    destdRepQQQ = os.path.join(destm,
                               "{:02d}".format(calendar.monthrange(endRepYear, endRepMonth)[1]))
    if not (os.path.isdir(destdRepQQQ)):
        os.mkdir(destdRepQQQ)

    desty = os.path.join(Q95Fold, "{:04d}".format(endRepYear))
    if not (os.path.isdir(desty)):
        os.mkdir(desty)
    destm = os.path.join(desty, "{:02d}".format(endRepMonth))
    if not (os.path.isdir(destm)):
        os.mkdir(destm)
    destdRepQ95 = os.path.join(destm,
                               "{:02d}".format(calendar.monthrange(endRepYear, endRepMonth)[1]))
    if not (os.path.isdir(destdRepQ95)):
        os.mkdir(destdRepQ95)

    # create database connection
    print("Trying to connect to the database : ", dbFileName)
    conn = new_db_connection(dbFileName)
    if conn is not None:
        print("connection to database is established successfully..")

    # get list of all stations
    print("getting list of all stations")
    sqlStr = "SELECT Id_Station FROM Debits"
    codes = fetch_db_data(conn, sqlStr)
    codes = get_unique_list(codes)

    # read statistics (Q95 table)
    with open(q95Table, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        codes = get_valid_list(codes, csv_reader)
    print("total number of stations is ", len(codes))
    
    # selection SQL command (text)
    selection = "SELECT "
	
	# initialise lists
    latestCode = []
    latestStart = []
    latestEnd = []
    latestLFI = []
	
	# Cycle for each station (code)
    for code in codes:

        print("get data of station No. ", code)
        fromWhere = " FROM Stations_Base WHERE Id_Station=\'" + code + "\'"
        # Get station information
        name = fetch_db_data(conn, selection + "Nom" + fromWhere)[0][0]
        river = fetch_db_data(conn, selection + "Riviere" + fromWhere)[0][0]
        x = fetch_db_data(conn, selection + "Longitude" + fromWhere)[0][0]
        y = fetch_db_data(conn, selection + "Latitude" + fromWhere)[0][0]
        z = fetch_db_data(conn, selection + "Altitude" + fromWhere)[0][0]
        sq = sqClass(code, x, y, z, name, river)
        fromWhere = "  FROM Debits WHERE Id_Station=\'" + code + "\'"
        # Get time boundaries
        Time = fetch_db_data(conn, selection + "Date" + fromWhere)
        Data = fetch_db_data(conn, selection + "Valeur" + fromWhere)
        select_Time = []
        select_Data = []
        tstart = datetime(startYear, startMonth, 1).toordinal()
        mrange = calendar.monthrange(endYear, endMonth)
        tend = datetime(endYear, endMonth, mrange[1]).toordinal()
        tstartRep = datetime(startRepYear, startRepMonth, 1).toordinal()
        tstartRepT = datetime(startRepYear, startRepMonth, 1)
        mrange = calendar.monthrange(endRepYear, endRepMonth)
        tendRep = datetime(endRepYear, endRepMonth, mrange[1]).toordinal()
        tendRepT = datetime(endRepYear, endRepMonth, mrange[1])
        # select data in time range
        for tt, dd in zip(Time, Data):
            tdata = datetime.strptime((tt[0]), '%Y-%m-%d %H:%M:%S').date().toordinal()
            if tstart <= tdata <= tend:
                select_Time.append(tt)
                select_Data.append(dd)
        sq.set_data(select_Data, select_Time)
        # check density (enough data in time range)
        if sq.has_enough_data(startYear, endYear, mdens):
            sq.inherit_DC(q95Table)
            sq.find_events(tstartRep, tendRep)
            # create output
            create_csv_events(sq, evFolder)
            create_reports(sq, destd, destdRepQQQ, destdRepQ95, tstartRepT, tendRepT)
            latestCode.append(sq.code)
            if len(sq.tStartev) > 0:
                latestStart.append(sq.tStartev[-1])
                latestEnd.append(sq.tEndev[-1])
                latestLFI.append(1 - numpy.exp(- sq.lambd_ev * sq.dVev[-1]))
            else:
                latestStart.append("0000-00-00")
                latestEnd.append("0000-00-00")
                latestLFI.append(-99999)
    latestTable = os.path.join(destd, "LFI-latest_{:04d}{:02d}010000.csv".format(endYear, endMonth))
    create_csv_latest(latestTable, latestCode, latestStart, latestEnd, latestLFI)



