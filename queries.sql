-- check the raw data
SELECT is_aggressive, COUNT(*)
GROUP BY is_aggressive;

SELECT activity_id,COUNT(*) 
GROUP BY activity_id;

-- Neuromuscular feature profiling
SELECT 
	is_aggressive,
	ROUND(AVG(ABS(r_bicep)), 4) AS avg_r_bicep,
	ROUND(AVG(ABS(r_tricep)), 4) AS avg_r_tricep,
	ROUND(AVG(ABS(l_bicep)), 4) AS avg_l_bicep,
	ROUND(AVG(ABS(l_tricep)), 4) AS avg_l_tricep,
	ROUND(AVG(ABS(r_thigh)), 4) AS avg_r_thigh,
	ROUND(AVG(ABS(r_hamstring)), 4) AS avg_r_hamstring,
	ROUND(AVG(ABS(l_thigh)), 4) AS avg_l_thigh,
	ROUND(AVG(ABS(l_hamstring)), 4) AS avg_l_hamstring
FROM emg_data
GROUP BY  is_aggressive;