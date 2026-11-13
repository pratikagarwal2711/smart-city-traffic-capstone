/*
Tasks:

Task 1: SQL-Based Traffic Analysis
Using the Metro Interstate Traffic Volume dataset:
1.1 Load the dataset
•	Load the comma-delimited traffic CSV file into SQLite. 
•	Verify that the data has been loaded correctly. 

1.2 Analyse annual traffic trends
Compare the total yearly traffic volume for:
•	2012 
•	2013 
•	2014 
•	2015 
•	2016 
•	2017 

Then:
•	Identify years with increases or decreases. 
•	Calculate the change between years where appropriate. 
•	Explain at least two meaningful observations about traffic volume trends. 
*/

with raw as (
SELECT SUBSTR(date_time,INSTR(date_time, ' ') - 4, 4) AS year, date_time, traffic_volume
FROM Metro_Interstate_Traffic_Volume)
SELECT year, count(date_time), sum(traffic_volume), sum(traffic_volume) / count(date_time) as average
from raw
where year in ('2012','2013','2014','2015','2016','2017')
group by year;

/* average indicates the increase / decrease. 2012 - baseline, 2013 - increase, 2014 - decrease, 2015 - decrease, 2016 - increase, 2017 - increase

OBSERVATIONS:
1. The large swings in total volume come mostly from missing data, not demand. While total volume shows a large swing over the year, its mostly due to number of data points made available
2. Underlying traffic was roughly flat from 2012 to 2016, then grew in 2017. This is evident from the average numbers

*/

/*
1.3 Analyse temperature around holidays
Compare temperature patterns for 2015, 2016 and 2017 during:
•	New Year's Day 
•	Labor Day 
Identify notable year-on-year changes and discuss whether these changes appear relevant to traffic conditions.

*/


SELECT holiday, date_time, SUBSTR(date_time,INSTR(date_time, ' ') - 4, 4) AS year  FROM Metro_Interstate_Traffic_Volume
WHERE holiday IN ('New Years Day','Labor Day') 
AND SUBSTR(date_time,INSTR(date_time, ' ') - 4, 4) in ('2015','2016','2017')
ORDER BY date_time;

-- Step 1: the holiday dates (about 12 rows)
CREATE TEMP TABLE hol AS
SELECT DISTINCT holiday,
       substr(date_time, 1, instr(date_time,' ')-1)              AS d,
       CAST(substr(date_time, instr(date_time,' ')-4, 4) AS INT) AS yr
FROM Metro_Interstate_Traffic_Volume
WHERE holiday IN ('New Years Day','Labor Day');

-- Step 2: only the hours on those dates (one row per hour, in Celsius)
CREATE TEMP TABLE hol_hours AS
SELECT hol.holiday, hol.yr, hol.d, m.date_time,
       CAST(m.temp AS REAL) - 273.15   AS temp_c,
       CAST(m.traffic_volume AS INT)   AS traffic_volume,
       MIN(m.rowid)                    AS keep_row      -- removes repeated hours
FROM hol
JOIN Metro_Interstate_Traffic_Volume m ON m.date_time LIKE hol.d || ' %'
WHERE CAST(m.temp AS REAL) > 0
GROUP BY hol.holiday, hol.yr, hol.d, m.date_time;

-- Step 3: the result
WITH grid AS (
  SELECT h.holiday, y.yr
  FROM (SELECT 'New Years Day' AS holiday UNION ALL SELECT 'Labor Day') h
  CROSS JOIN (SELECT 2015 AS yr UNION ALL SELECT 2016 UNION ALL SELECT 2017) y),
daily AS (
  SELECT holiday, yr, d AS holiday_date,
         COUNT(*)                       AS hours,
         ROUND(AVG(temp_c),1)           AS avg_temp_c,
         ROUND(MIN(temp_c),1)           AS min_temp_c,
         ROUND(MAX(temp_c),1)           AS max_temp_c,
         ROUND(AVG(traffic_volume),0)   AS avg_traffic
  FROM hol_hours
  WHERE yr BETWEEN 2015 AND 2017
  GROUP BY holiday, yr, d)
SELECT g.holiday, g.yr, d.holiday_date, d.hours,
       d.avg_temp_c, d.min_temp_c, d.max_temp_c, d.avg_traffic,
       ROUND(d.avg_temp_c - LAG(d.avg_temp_c)
             OVER (PARTITION BY g.holiday ORDER BY g.yr), 1) AS change_vs_prior_year_c
FROM grid g
LEFT JOIN daily d ON d.holiday = g.holiday AND d.yr = g.yr
ORDER BY g.holiday DESC, g.yr;

/*
New Year’s Day was much warmer in 2017. The holiday day was 4.8°C warmer than in 2016
Labor Day was stable in 2015 and 2016, then clearly cooler in 2017.

Do these changes appear relevant to traffic?
Based only on the query result, the temperature changes do not show a consistent link to traffic.
New Year’s Day: Traffic rose by about 14% when the temperature rose by 4.8°C. Taken alone, this fits warmer weather coinciding with more traffic.
Labor Day 2015 → 2016: The temperature was almost unchanged (−0.3°C), yet traffic fell by 11.4%. Something other than temperature caused that drop.
Labor Day 2016 → 2017: The temperature fell by 3.8°C, yet traffic rose by 11.3%. This is the opposite of what a “cooler means less traffic” explanation would predict.

Conclusion: Only one of the three comparisons is consistent with temperature affecting traffic. 
The other two show traffic changing by about 11% when temperature either barely moved or moved the opposite way. 
The temperature changes therefore do not appear relevant to traffic conditions in this result. 
Traffic changes between years look driven by other factors that this query does not measure.

*/


