# Compute Low Flow Index statistics (Q95), using SENAMHI ground stations DB 
# Developed at Fondazione CIMA, 2020
# for Python 3.7

# Configuration file needed: LFI_config.json

# Load required Libraries
import os
# from logging.config import stopListening

import csv
import scipy.stats as stats
import jaydebeapi
from datetime import datetime
from datetime import date as dateFunc
from datetime import timedelta
import operator
import numpy
import pyodbc
import json

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

# Create q95 table (header)
def create_header_row_q95():
    row = []
    row.append("x")
    row.append("y")
    row.append("z")
    row.append("code")
    row.append("name")
    row.append("river")
    row.append("start")
    row.append("end")
    row.append("dens")
    for month in range(1, 13):
        row.append("Q95_" + ("%02d" % month))
    row.append("lambda")
    return row

# create q95 table (values)
def create_data_row_q95(sObj):
    row = []
    # info
    row.append(sObj.x)
    row.append(sObj.y)
    row.append(sObj.z)
    row.append(sObj.code)
    row.append(sObj.name)
    row.append(sObj.river)
    row.append(sObj.start)
    row.append(sObj.end)
    row.append(sObj.dens)
    # value
    for month in range(1, 13):
        row.append(("%.3f" % sObj.DC[month-1][0]))
    row.append(sObj.lambd_ev)
    # print(row)
    return row

# Create csv table of q95 values, one row for each station
def create_csv_q95(fileName, sList):
    with open(fileName, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow(create_header_row_q95())
        for sObj in sList:
            dataRow = []

            dataRow = create_data_row_q95(sObj)

            writer.writerow(dataRow)

# Search for low flow events and create csv report, from sq object (per station)
def create_csv_events(sq, evFolder):
    fileName = os.path.join(evFolder, sq.code + "-LowFlowEvents.csv")
    with open(fileName, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        row = []
        row.append("Start")
        row.append("End")
        row.append("Dvol")
        row.append("LFI")
        writer.writerow(row)
        for tstart, tend, dV in zip(sq.tStartev, sq.tEndev, sq.dVev):
            row = []
            row.append(tstart)
            row.append(tend)
            row.append(dV)
            row.append(1 - numpy.exp(- sq.lambd_ev * dV))
            writer.writerow(row)
            # print(row)

if __name__ == '__main__':
    # start main program
    print("Hello world")

	# Read configuration file (in same folder)
    jsoname = 'LFI_config.json'
    with open(jsoname) as jf:
        params = json.load(jf)
        dbFileName = params["DataBase"]
        mstart = params["MaxStartYear"]
        mend = params["MinEndYear"]
        mdens = params["MinDataDens"]
        q95Table = params["Q95Table"]
        evFolder = params["EventTablesFolder"]
        if not (os.path.isdir(evFolder)):
            os.mkdir(evFolder)

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
    print("total number of stations is ", len(codes))
    sqList = []
    
    # selection SQL command (text)
    selection = "SELECT "
	
	# Cycle for each station (code)
    for i in codes:

        code = i[0]
        print("get data of station No. ", code)
        fromWhere = " FROM Stations_Base WHERE Id_Station=\'" + code + "\'"
        # Get station info
        name = fetch_db_data(conn, selection + "Nom" + fromWhere)[0][0]
        river = fetch_db_data(conn, selection + "Riviere" + fromWhere)[0][0]
        x = fetch_db_data(conn, selection + "Longitude" + fromWhere)[0][0]
        y = fetch_db_data(conn, selection + "Latitude" + fromWhere)[0][0]
        z = fetch_db_data(conn, selection + "Altitude" + fromWhere)[0][0]
        sq = sqClass(code, x, y, z, name, river)
        fromWhere = "  FROM Debits WHERE Id_Station=\'" + code + "\'"
        # Get data and time
        Time = fetch_db_data(conn, selection + "Date" + fromWhere)
        Data = fetch_db_data(conn, selection + "Valeur" + fromWhere)
        sq.set_data(Data, Time)
        # if enough data, save
        if sq.has_enough_data(mstart, mend, mdens):
            sq.set_DC()
            sqList.append(sq)

    for sq in sqList:
        sq.find_events(datetime(1900, 1, 1).toordinal(), datetime(2100, 1, 1).toordinal())
        create_csv_events(sq, evFolder)
	
	# Create csv table
    create_csv_q95(q95Table, sqList)
