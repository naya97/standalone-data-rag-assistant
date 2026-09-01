-- Question 1: Top 5 orders by sales
SELECT Order_ID, Customer_Name, Sales
FROM orders
ORDER BY Sales DESC
LIMIT 5;


-- Question 2: How many orders belong to the Furniture category
SELECT COUNT(*) AS Count
FROM orders
WHERE Category = 'Furniture';


-- Question 3: Total sales by region
SELECT Region, SUM(Sales) AS Total_Sales
FROM orders
GROUP BY Region
ORDER BY Total_Sales DESC;


-- Question 4: Average profit by sub-category
SELECT Sub_Category, AVG(Profit) AS Avg_Profit
FROM orders
GROUP BY Sub_Category
ORDER BY Avg_Profit DESC;


-- Question 5: Percentage of losing orders (Profit < 0)
SELECT
    (SUM(CASE WHEN Profit < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS Loss_Percentage
FROM orders;


-- Question 6: Relation between discount level and average profit
SELECT Discount, AVG(Profit) AS Avg_Profit
FROM orders
GROUP BY Discount
ORDER BY Discount ASC;


-- Question 7: Average shipping days by ship mode
SELECT
    Ship_Mode,
    AVG(julianday(Ship_Date) - julianday(Order_Date)) AS Avg_Shipping_Days
FROM orders
GROUP BY Ship_Mode
ORDER BY Avg_Shipping_Days ASC;


-- Question 8: Most frequent customer segment (by order count)
SELECT Segment, COUNT(*) AS Order_Count
FROM orders
GROUP BY Segment
ORDER BY Order_Count DESC;


-- Question 9: Total sales by year
SELECT Order_Year, SUM(Sales) AS Total_Sales
FROM orders
GROUP BY Order_Year
ORDER BY Order_Year;


-- Question 10: Month with the highest number of orders
SELECT Order_Month, COUNT(*) AS Order_Count
FROM orders
GROUP BY Order_Month
ORDER BY Order_Count DESC;


--  Get-Content SQL/superstore_queries.sql | sqlite3 data/db/superstore.db