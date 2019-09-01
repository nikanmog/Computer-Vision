# Import all necessary packages
import sqlite3, datetime, platform

# Open Connection to database
conn = sqlite3.connect('util/Measurements')
run = datetime.datetime.now()
c = conn.cursor()
measurement = 0


# Save a measurement to the sqlite database
def persist(approach:str, duration, confidence=0.0):
    global measurement
    measurement += 1
    c.execute('INSERT INTO performance values (?,?,?,?,?,?)', (run, duration, approach, platform.system(), measurement, confidence))


# Close Database Connection
def close():
    conn.commit()