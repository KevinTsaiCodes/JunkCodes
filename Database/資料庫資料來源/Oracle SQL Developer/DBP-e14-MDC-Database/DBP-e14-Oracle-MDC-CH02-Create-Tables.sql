/******************************************************************************/
/*									                                                        	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02   			  */
/*										                                                        */
/*	The Marcia's Dry Cleaing (MDC-CH02) Database - Create Tables              */
/*							                                                              */
/*	These are the Oracle Database 11g R2 and 12c SQL code solutions	                  */
/*								                                                            */
/******************************************************************************/


CREATE TABLE CUSTOMER (
		CustomerID		Int				NOT NULL,
		FirstName 		Char(25)		NOT NULL,
		LastName			Char(25)		NOT NULL,
		Phone				  Char(12)		NOT NULL,
		Email				  Char(100)		NULL,
		CONSTRAINT		Customer_PK	PRIMARY KEY (CustomerID)
		);


CREATE TABLE INVOICE (
		InvoiceNumber		Int				NOT NULL,
		CustomerNumber		  Int				NOT NULL,
		DateIn				  Date			NOT NULL,
		DateOut				  Date			NOT NULL,
		TotalAmount				    Number(8,2)	NULL,
		CONSTRAINT		InvoicePK		PRIMARY KEY (InvoiceNumber),
		CONSTRAINT  		Invoice_Cust_FK FOREIGN KEY(CustomerNumber)
								REFERENCES CUSTOMER(CustomerID)
		);

CREATE TABLE INVOICE_ITEM (
		InvoiceNumber		Int				NOT NULL,
		ItemNumber		  Int				NOT NULL,
		Item			    Char(50)				NOT NULL,
    Quantity			  Int				NOT NULL,
		UnitPrice			  Number(8,2)	NULL,
		CONSTRAINT		Invoice_ItemPK	
								PRIMARY KEY(InvoiceNumber, ItemNumber),
		CONSTRAINT		Item_Invoice_FK	FOREIGN KEY(InvoiceNumber)
								REFERENCES INVOICE(InvoiceNumber)
									ON DELETE CASCADE
		);

/******************************************************************************/



