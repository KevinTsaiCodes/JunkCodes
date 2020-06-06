/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			*/
/*																				*/
/*	The Queen Anne Curiousity Shop Database Create Tables						*/
/*																				*/
/*	These are the Microsoft SQL Server 2012 / 2014 SQL code solutions			*/
/*																				*/
/********************************************************************************/
/*																				*/
/*	NOTE:  These SQL Statements use the following surrogate keys:				*/
/*																				*/
/*		CUSTOMER:	Start at 1, Increment by 1 - IDENTITY (1, 1) 				*/
/*		ITEM:		Start at 1, Increment by 1 - IDENTITY (1, 1)				*/
/*		SALE:		Start at 1, Increment by 1 - IDENTITY (1, 1)				*/
/*																				*/
/********************************************************************************/

CREATE  TABLE CUSTOMER(
	CustomerID			Int				NOT NULL IDENTITY (1, 1),
	LastName			Char(25)		NOT NULL,
	FirstName			Char(25)		NOT NULL,
	[Address]			Char(35)		NULL,
	City				Char(35)		NULL,
	[State]				Char(2)			NULL,
	ZIP					Char(10)		NULL,
	Phone				Char(12)		NOT NULL,
	Email				VarChar(100)	NULL,
	CONSTRAINT 			CUSTOMER_PK 	PRIMARY KEY(CustomerID)
	);


/*****   ITEM Table As Used in Chapter02   **************************************/

CREATE  TABLE ITEM(
	ItemID				Int	 			NOT NULL IDENTITY (1,1),
	ItemDescription		VarChar(255)	NOT NULL,
	CompanyName			VarChar(100)	NOT NULL,
	PurchaseDate		Date			NOT NULL,
	ItemCost			Numeric(9,2)	NOT NULL,
	ItemPrice			Numeric(9,2)	NOT NULL,
	CONSTRAINT 			ITEM_PK			PRIMARY KEY (ItemID),
	);

CREATE  TABLE SALE(
    SaleID				Int				NOT NULL IDENTITY (1, 1),
	CustomerID			Int	 			NOT NULL,
	SaleDate			Date			NOT NULL,
	SubTotal			Numeric(15,2)	NULL,
	Tax					Numeric(15,2)	NULL,
	Total				Numeric(15,2)	NULL,
    CONSTRAINT 			SALE_PK 		PRIMARY KEY (SaleID),
	CONSTRAINT 			SALE_CUSTOMER_FK FOREIGN KEY (CustomerID)
								REFERENCES Customer(CustomerID)
										ON UPDATE NO ACTION
										ON DELETE NO ACTION,
	);

/*****   SALE_ITEM Table As Used in Chapter02   *********************************/

CREATE  TABLE SALE_ITEM(
	SaleID				Int				NOT NULL,
	SaleItemID			Int				NOT NULL,
	ItemID				Int	 			NOT NULL,
	ItemPrice			Numeric(9,2)	NOT NULL,
    CONSTRAINT 			SALE_ITEM_PK 	PRIMARY KEY (SaleID, SaleItemID),
	CONSTRAINT 			SALE_ITEM_SALE_FK FOREIGN KEY (SaleID)
								REFERENCES SALE(SaleID)
 										ON UPDATE NO ACTION
 										ON DELETE CASCADE,
 	CONSTRAINT 			SALE_ITEM_ITEM_FK FOREIGN KEY (ItemID)
 								REFERENCES ITEM(ItemID)
 										ON UPDATE NO ACTION
 										ON DELETE NO ACTION
    );
 

/********************************************************************************/
