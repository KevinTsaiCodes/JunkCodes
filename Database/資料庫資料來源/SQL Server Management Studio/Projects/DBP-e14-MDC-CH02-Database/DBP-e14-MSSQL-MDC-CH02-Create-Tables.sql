/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			*/
/*																				*/
/*	Marcia's Dry Cleaning [MDC-CH02] Project Create Tables						*/
/*																				*/
/*	Microsoft SQL Server 2012 /2014 code solutions`								*/
/********************************************************************************/

CREATE TABLE CUSTOMER(
		CustomerID			Int				NOT NULL IDENTITY (1, 1),
		FirstName 			Char(25)		NOT NULL,
		LastName			Char(25)		NOT NULL,
		Phone				Char(12)		NOT NULL,
		Email				Char(100)		NULL,
		CONSTRAINT			CustomerPK		PRIMARY KEY(CustomerID)
		);


CREATE TABLE INVOICE(
		InvoiceNumber		Int				NOT NULL IDENTITY (2015001, 1),
		CustomerNumber		Int				NOT NULL,
		DateIn				Date			NOT NULL,
		DateOut				Date			NULL,
		TotalAmount			Numeric(8,2)	NULL,
		CONSTRAINT			InvoicePK			PRIMARY KEY (InvoiceNumber),
		CONSTRAINT  		Invoice_Cust_FK 	FOREIGN KEY(CustomerNumber)
							REFERENCES CUSTOMER(CustomerID)
								ON UPDATE NO ACTION
								ON DELETE NO ACTION
		);

CREATE TABLE INVOICE_ITEM(
		InvoiceNumber	Int				NOT NULL,
		ItemNumber		Int				NOT NULL,
		Item			Char(50)		NOT NULL,
		Quantity		Int				NOT NULL DEFAULT 1,
		UnitPrice		Numeric(8,2)	NULL,
		CONSTRAINT		Invoice_ItemPK	PRIMARY KEY(InvoiceNumber, ItemNumber),
		CONSTRAINT		Item_Invoice_FK	FOREIGN KEY(InvoiceNumber)
							REFERENCES INVOICE(InvoiceNumber)
								ON UPDATE CASCADE
								ON DELETE CASCADE,
		);

/********************************************************************************/





