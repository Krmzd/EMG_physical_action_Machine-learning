CREATE DATABASE emg_action;
USE emg_action;
CREATE TABLE emg (
 ID INT auto_increment PRIMARY KEY,
 r_bicep FLOAT,
 r_tricep FLOAT, 
 l_bicep FLOAT, 
 l_tricep FLOAT, 
 r_thigh FLOAT, 
 r_hamstring FLOAT, 
 l_thigh FLOAT, 
 l_hamstring FLOAT, 
 is_aggressive INT
 );

 