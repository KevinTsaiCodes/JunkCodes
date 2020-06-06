/******************************************************************************/
/*																				                                    */
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			    */
/*																				                                    */
/*	Cape Codd Outdoor Sports - Create Tables									                */
/*																				                                    */
/*	These are the ORACLE Database SQL code solutions		                    	*/
/*																				                                    */
/******************************************************************************/

CREATE TABLE RETAIL_ORDER (
	OrderNumber     Integer 	          NOT NULL,
	StoreNumber     Integer 	          NULL,
	StoreZip        Char(9) 	          NULL,
	OrderMonth    	Char(12) 	          NOT NULL,
	OrderYear     	Integer 	          NOT NULL,
	OrderTotal   	  Number(8,2)         NULL,
	CONSTRAINT 		  RETAIL_ORDER_PK     PRIMARY KEY (OrderNumber)
	);

CREATE TABLE SKU_DATA (
	SKU   			    Integer 	          NOT NULL,
	SKU_Description Char(35) 	          NOT NULL,
	Department  	  Char(30)  	        NOT NULL,
	Buyer  			    Char(35)  	        NULL,
	CONSTRAINT 		  SKU_DATA_PK 	      PRIMARY KEY (SKU)
	);

CREATE TABLE ORDER_ITEM (
	OrderNumber  	  Integer  	          NOT NULL,
	SKU  			      Integer     	      NOT NULL,
	Quantity  		  Integer 	          NOT NULL,
	Price  			    Number(8,2)	 	      NOT NULL,
	ExtendedPrice  	Number(8,2)		      NOT NULL,
	CONSTRAINT 		  ORDER_ITEM_PK       PRIMARY KEY (SKU, OrderNumber),
	CONSTRAINT 		  SKU_Relationship 	  FOREIGN KEY (SKU)
                          REFERENCES 	SKU_DATA (SKU),
	CONSTRAINT 		  RETAIL_ORDER_Relationship 	FOREIGN KEY (OrderNumber)
                          REFERENCES 	RETAIL_ORDER (OrderNumber)
	);
	
CREATE TABLE WAREHOUSE (
	WarehouseID  		Integer       NOT NULL,
  WarehouseCity   Char(30)  	  NOT NULL,
	WarehouseState  Char(2)       NOT NULL,
  Manager			    Char(35)	    NULL,
	SquareFeet		  Integer		    NULL,
	CONSTRAINT 		  WAREHOUSE_PK 	      PRIMARY KEY (WarehouseID)
	);
	
CREATE TABLE INVENTORY (
	WarehouseID     Integer       NOT NULL,
  SKU   			    Integer 	    NOT NULL,
	SKU_Description	Char(35)    	NOT NULL,
	QuantityOnHand 	Integer  	    NULL,
	QuantityOnOrder	Integer  	    NULL,
	CONSTRAINT 		  INVENTORY_PK 	      PRIMARY KEY (WarehouseID, SKU),
	CONSTRAINT 		  SKU_DATA_Relationship   FOREIGN KEY (SKU)
                          REFERENCES 	SKU_DATA (SKU),
	CONSTRAINT 		  WAREHOUSE_Relationship 	FOREIGN KEY (WarehouseID)
                          REFERENCES 	WAREHOUSE (WarehouseID)
	);
  
  CREATE TABLE CATALOG_SKU_2013 (
	CatalogID		Integer		NOT NULL,
	SKU   			Integer 	NOT NULL,
	SKU_Description Char(35) 	NOT NULL,
	Department  	Char(30)  	NOT NULL,
	CatalogPage		Integer		NULL,
	DateOnWebSite	Date		NULL,
	CONSTRAINT 		CSKU13_PK 	PRIMARY KEY (CatalogID)
	);
  
CREATE SEQUENCE seqCSKU13 INCREMENT BY 1 START WITH 20130001;

CREATE TABLE CATALOG_SKU_2014 (
	CatalogID		Integer		NOT NULL,
	SKU   			Integer 	NOT NULL,
	SKU_Description Char(35) 	NOT NULL,
	Department  	Char(30)  	NOT NULL,
	CatalogPage		Integer		NULL,
	DateOnWebSite	Date		NULL,
	CONSTRAINT 		CSKU14_PK 	PRIMARY KEY (CatalogID)
	);
  
CREATE SEQUENCE seqCSKU14 INCREMENT BY 1 START WITH 20140001;

CREATE TABLE CATALOG_SKU_2015 (
	CatalogID		Integer		NOT NULL,
	SKU   			Integer 	NOT NULL,
	SKU_Description Char(35) 	NOT NULL,
	Department  	Char(30)  	NOT NULL,
	CatalogPage		Integer		NULL,
	DateOnWebSite	Date		NULL,
	CONSTRAINT 		CSKU15_PK 	PRIMARY KEY (CatalogID)
	);

CREATE SEQUENCE seqCSKU15 INCREMENT BY 1 START WITH 20150001;
