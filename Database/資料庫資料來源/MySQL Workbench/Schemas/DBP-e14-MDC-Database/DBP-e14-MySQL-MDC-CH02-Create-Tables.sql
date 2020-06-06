/********************************************************************************/
/*										                                      	*/
/*	Kroenke and Auer - Database Processing (14th Edition) ch 2					*/
/*										                                      	*/
/*	Marcia's Dry Cleaning [MDC] Project Create Tables	                    	 */
/*													                        	 */
/*	These are the MySQL 5.6 SQL code solutions		                          	*/
/*  Note: MySQL does not support auto_increment with a step besides 1. 			*/
/*  	Thus, customerIDs are inserted manually 100, 105, 110, etc. 			*/		
/********************************************************************************/

CREATE TABLE CUSTOMER(
		CustomerID		Int				    NOT NULL auto_increment,
		FirstName 		Char(25)	    NOT NULL,
		LastName		  Char(25)	    NOT NULL,
		Phone			    Char(12) 	    NOT NULL,
		Email			    varchar(100)	    NULL,
		CONSTRAINT		CustomerPK    PRIMARY KEY(CustomerID)
		);



CREATE TABLE INVOICE(
		InvoiceNumber       Int				   NOT NULL auto_increment,
		CustomerNumber      Int				   NOT NULL,
		DateIn			        Date	   	   NOT NULL,
		DateOut			        Date  	 	   NULL,
		TotalAmount		   Numeric(8,2)	NULL,
		CONSTRAINT		InvoicePK			PRIMARY KEY (InvoiceNumber),
		CONSTRAINT  	Invoice_Cust_FK 	FOREIGN KEY(CustomerNumber)
							REFERENCES CUSTOMER(CustomerID)
                 		);

ALTER TABLE invoice AUTO_INCREMENT = 20150001; 

CREATE TABLE INVOICE_ITEM(
		InvoiceNumber   Int				    NOT NULL,
		ItemNumber		Int				    NOT NULL,
		Item			Char(50)			NOT NULL,
		Quantity        Int				    NOT NULL DEFAULT 1,
		UnitPrice		Numeric(8,2)		NULL,
		CONSTRAINT		InvoiceItemPK	PRIMARY KEY(InvoiceNumber, ItemNumber),
		CONSTRAINT		Invoice_Item_FK	FOREIGN KEY(InvoiceNumber)
							REFERENCES INVOICE(InvoiceNumber)
								ON UPDATE CASCADE
								ON DELETE CASCADE
		    );

/********************************************************************************/




