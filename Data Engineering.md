# 1. Data Modeling

- Procedure
  - Gather requirements: define the business requirements
  - Conceptual data modeling: determine the entity and attributes
  - Logical data modeling: define the data structure
  - Physical data modeling: define the relationship between entity, such as primary key & foreign key

# 2. Relational Database

- ACID: atomicity，consistency, isolation, durability
  - Atomicity: （不可分割性）
  - Consistency: (一致性)
  - Isolation: （隔离性 事务之间是隔离的， 一个事务不影响其他事物的运行）
  - Durability: （持久性 更改永久的保存在数据库中 不会被回滚）
  
- When not to use relational database:
  - Large amount of data
  - Need to be able to store different data type format
  - Need high throughput - fast reads
  - Need flexible schema
  - Need High availablity
  - Need horizontal scalability

- OLTP vs OLAP:
  - OLTP: online transactional processing
  - OLAP: online analytical processing

- Normalization vs Denormalization:
  - Normalization: to reduce data redundancy and increase data integrity
    - Normalization rules divides larger tables into smaller tables and links them using relationships.
  - Denormalization: must be done in read heavy workloads to increase performance

- Normal Forms
  - 1NF: every attribute cannot be divided (no sets/ lists)
  - 2NF: based on 1NF, remove the nonprimary attribute’s partial functional dependency on key(must rely on primary key)
  - 3NF: based on 2NF, remove the non primary attribute’s dependency on key(no transitive dependencies)

- 6NF
  - (1）1NF: each table cell should contain a single value | Each record needs to be unique
  - (2) 2NF: single column primary key
  - (3) 3NF: has no transitive functional dependencies
  - (4) 4NF: If no database table instance contains two or more, independent and multivalued data describing the relevant entity, then it is in 4th Normal Form.
  - (5)  5NF: A table is in 5th Normal Form only if it is in 4NF and it cannot be decomposed into any number of smaller tables without loss of data.
  - (6)  6NF: 6th Normal Form is not standardized, yet however, it is being discussed by database experts for some time. Hopefully, we would have a clear & standardized definition for 6th Normal Form in the near future...

- Denormalization:  Logical Design Change

- Fact & Dimension Tables
  - Fact tables: consists of the measurements, metrics or facts of a business process
  - Dimension tables: a structure that categorizes facts and measures in order to enable users to answer the business questions

- Star schema & snowflake
  - Star: consists of one of more fact tables referencing any number of dimension tables
  - Snowflake: multiple dimension tables referencing. The dimension tables are normalized which splits data into additional tables.
  - galaxy: contains two fact table that share dimension tables between them

- Slowly Changing Dimension
  - Data Warehouse there is a need to track changes in dimension attributes in order to report historical data
  - Type 0 - The passive method
  - Type 1 - Overwriting the old value
  - Type 2 - Creating a new additional record, using flag to note if it is the current status
  - Type 3 - Adding a new column, current_type/ previous_type
  - Type 4 - Using historical table, mark the effective start_date and end_date
  - Type 6 - Combine approaches of types 1,2,3 (1+2+3=6)

- key type:
  - primary key: used to ensure data in the specific column is unique
  - foreign key: a column or group of columns in a relational database table that provides a link between data in two tables
  - surrogate key:  a made up value with the sole purpose of uniquely identifying a row. Usually, this is represented by an auto incrementing ID(system generated)
  - nature key: A natural key is a column or set of columns that already exist in the table (e.g. they are attributes of the entity within the data model) and uniquely identify a record in the table

- view: the result of the query

# 3. NoSQL - not only SQL

- what is nosql
  - sql is relational, no-sql is non-relational
  - nosql has dynamic schemas for unstructured data
  - nosql is document, key-value, graph or wide-columnn stores while sql are table based
  - sql database are better for multi-row transactions

- When to use noSQL
  - Need high availability
  - Need high Availability in the data: Indicates the system is always up and there is no downtime
  - Have Large Amounts of Data
  - Need Linear Scalability: The need to add more nodes to the system so performance will increase linearly
  - Low Latency: Shorter delay before the data is transferred once the instruction for the transfer has been received.
  - Need fast reads and write

- Eventual consistency: if no new updates are made to a given data item., eventually all accesses to that item will return the last updated value

- CAP Theorem:
  - Consistency: Every read from the database gets the latest (and correct) piece of data or an error(所有节点在同一时间具有相同的数据)
  - Availability: Every request is received and a response is given -- without a guarantee that the data is the latest update(保证每个请求不管失败或者成功都有响应)
  - Partition Tolerance: The system continues to work regardless of losing network connectivity between nodes(系统中任意信息的丢失或失败不回影响系统的继续运行)

- Key Type:
  - Primary key: how each row can be uniquely identified and how the data is distributed across the nodes in our system. The first element of the primary key is the partition key. Made up of either just the partition key or with the addition of clustering columns
  - partition key: The partition key is used for partitioning the data. Data with the same partition key is stored together, which allows you to query data with the same partition key in 1 query. hashed value is used to determine the node to store the data
  - Clustering columns: Any fields listed after the partition key are called clustering columns. These store data in ascending or descending order within the partition for the fast retrieval of similar values. All the fields together are the primary key.
  - composite key:If those fields are wrapped in parentheses then the partition key is composite.
  - sort key(redshift): Redshift Sort Key determines the order in which rows in a table are stored. Query performance is improved when Sort keys are properly used as it enables the query optimizer to read fewer chunks of data filtering out the majority of it.
  - distkey(redshift): determine where data is stored in Redshift. Clusters store data fundamentally across the compute nodes. Query performance suffers when a large amount of data is stored on a single node.
  
- index: a way of sorting a number of records on multiple fields. Creating an index on a field in a table creates another data structure which holds the field value, and a pointer to the record it relates to.
- Where clause must be included to. Execute queries

# 4. Data lake & Data Warehouse Structure

- Data Mart:  focus on one department
- Data Lake: contains unstructured data & structured data

- CIF

- Hybrid Kimball Bus. & Inmon CIF

# 5. Distribution Style（redshift）

---

- Even
  - Fact table, dimension table:  Go give each server a row
  - High cost of join(shuffling)
- ALL
  - Dimension tables: Copy them for all cpu (broadcasting)
- Auto
  - Leave decision to Redshift
  - Small tables apply ‘all'
  - Large tables apply ‘even'
- Key
  - Assign the cpu based on the same foreign key in fact table
  - A skewed distribution
  - Distkey

- Sort Key(substitute for indexing)
  - If you use ‘order by’ very frequently
- dist key

# 6. Data  pipeline - data  validation

- After loading from S3 to redshift
  - Validate the number of rows in redshift match the number of records in S3
- Once location business analysis is complete
  - Validate that all locations have a daily visit average greater than 0
  - Validate that the number locations in our output table math the number of tables in the input table

# 7. hadoop & mapreduce & hive

- mapreduce: A framework that helps programs do the parallel computation on data. The map task takes input data and converts it into a dataset that can be computed in key value pairs. The output of the map task is consumed by reduce tasks to aggregate output and provide the desired result. map是映射，reduce可以理解成归并映射或者说aggregation做聚合
- hadoop:
- hive: datawarehouse based on hadoop framework , the underlayer is HDFS, translate SQL to mapreduce to compute data
- HBase: Hadoop database, nosql database, mainly foucs on realtime query on big data

# 8. SQL

- indexinng:
  - data structures which hold field values from the indexed column(s) and pointers to the related record(s): allow rapid access to data
- partition:
  - data structures which hold field values from the indexed column(s) and pointers to the related record(s)
- Union vs union all:
  - union return distinct rows but slower
- drop vs delete vs truncate:
  - truncate is a DDL language, which delete all rows and it doesn't allow to use where;
  - drop will delete the data schema and the table won't exist;
  - delete is a DML language, you can specify the rows you want to remove
- how to delete duplicate rows:
  - using group by and having count(*) > 1
  - row_number()
- execution order:
  - from -》 join -》where -> group by -> having -> select -> distinct -> order by -> limit
- random sampling from database

```
select
row_number() over() row
from table
where mod(row, 100) = 1
```

- Dealing with Time: datediff(interval,date1,date2)，datepart(interval,date1), getdate()
- Count(distinct)
- Sum(case when - then feature =1 else 0 end)
- Windows function: rank() dense_rank (has duplicates) row_number()

# 9.  Business Intelligence

1. ETL: Extract,Transform, Load
2. Fact: additive fact, non-additive fact, semi-additive fact
3. materialized view:
4. Cube: it’s data processing unit compromised of fact tables and dimensions from the data warehouse
5. A data cube helps us represent data in multiple dimensions. It is defined by dimensions and facts. The dimensions are the entities with respect to which an enterprise preserves the records.
6. Grain of fact: the level at which fact information is stored
7. Factless table: a fact table without measure
8. Slowly changing table: since we have some updated information, for example someone changed their name in the system.
9. partition key
10. Practice

---

- pivot

```
```

-- Pivot table with one row and five columns  
SELECT 'AverageCost' AS Cost_Sorted_By_Production_Days,
[0], [1], [2], [3], [4]  
FROM  
(SELECT DaysToManufacture, StandardCost
    FROM Production.Product) AS SourceTable  
PIVOT  
(  
AVG(StandardCost)  
FOR DaysToManufacture IN ([0], [1], [2], [3], [4])  
) AS PivotTable;

```

```

SELECT VendorID, Employee, Orders  
FROM
   (SELECT VendorID, Emp1, Emp2, Emp3, Emp4, Emp5  
   FROM pvt) p  
UNPIVOT  
   (Orders FOR Employee IN
      (Emp1, Emp2, Emp3, Emp4, Emp5)  
)AS unpvt;

```
```

- select  nth hightest

```
method 1:
- LIMIT n DESC
- LIMIT 1 ASC

method 2:
WHERE N-1 = (SELECT COUNT(DISTINCT ) FROM e2 WHERE e1.a < e2.a

Method 3  
With TABLE SELECT a = ROW_NUMBER() OVER ( ORDER BY DESC)
SELECT WHERE a = n
```

- fetch rows in a and not in b

```
- NOT IN
- EXCEPT
- LEFT JOIN, WHERE B IS NULL
```

- in vs exist

```
exist:
- Works on Virtual tables 
- Is used with co-related queries
- Exits comparison when match is found  
- Performance is comparatively FAST for larger resultset of subquery-baidu 1point3acres
- Can compare anything with NULLs (SELECT NULL is TRUE by using EXISTS)

in:
- Works on List result set
- Doesn’t work on subqueries resulting in Virtual tables with multiple columns  
- Compares every value in the result list
- Performance is comparatively SLOW for larger resultset of subquery
- Cannot compare NULLs
```

- char vs varchar vs

```
- CHAR
- fixed length, regardless of the string it holds (any remaining space in the field is padded with blanks)
      
- VARCHAR: variable length, takes up 1 byte per character, + 2 bytes to hold length information (holds only the characters you assign to it, ). check 1point3acres for more.

- TRADEOFFS: when store, choose VARCHAR because take less spaces; when index, choose CHAR because require less string manipulation and faster
```

- rank vs dense_rank()

```
- Only difference: where there is a “tie”
  
- RANK() will assign non-consecutive “ranks” to the values in the set (resulting in gaps between the integer ranking values when there is a tie
  
- DENSE_RANK() will assign consecutive ranks to the values in the set (so there will be no gaps between the integer ranking values in the case of a tie)
```

- join type

```
- INNER JOIN (default): returns all rows for which there is at least one match in BOTH TABLES
  
- LEFT JOIN: returns all rows from the left table, and the matched rows from the right table; i.e. the results will contain all records from the left table, even if the JOIN condition doesn’t find any matching records in the right table (NULL)
  
- RIGHT JOIN: returns all rows from the right table, and the matched rows from the left table; the exact opposite of a LEFT JOIN
  
- FULL JOIN: returns all rows for which there is a match in either of the TABLES; combines the effect of applying both a LEFT JOIN and RIGHT JOIN; its result is equivalent to performing a UNION of the result of LEFT and RIGHT outer queries
  
- CROSS JOIN: returns all records where each row from the first table is combined with each row from the second table (i.e., returns the Cartesian product of the sets of rows from the joined tables); (a) be specified using the CROSS JOIN syntax (“explicit join notation”) (b) FROM clause separated by commas without using a WHERE (“implicit join notation”)
```

- Find the duplicate

```
GROUP BY fields, HAVING COUNT(*) > 1
```

- Copy data from one table to another

```
INSERT INTO table2 (a, b, c...) 
SELECT a, b, c… FROM table 1;
```

- substring

```
SUBSTRING(decimal, 2) 

- take decimal parts and make it integer
```

- Find the median

```
Select category, productid, round(avg(quantity)) as med from
(Select category,
productid,
quantity,
row_num() over (partition by categoryid order by quantity asce) row_num1,
row_num() over (partition by categoryid order by quantity desc) row_num2 
From order)
Where row_num1 in (row_num2 + 1,rownum2 -1)
Group by category,productid

```

- How to use sql to generate distribution

```
- count unique users for product view for a given time

- product view 
```
