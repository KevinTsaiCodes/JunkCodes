/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			*/
/*																				*/
/*	Marcia's Dry Cleaning [MDC-CH02] Project Insert Data						*/
/*																				*/
/*	Microsoft SQL Server 2012 / 2014 code solutions								*/
/*																				*/
/********************************************************************************/

/*****  CUSTOMER Data   *********************************************************/

INSERT INTO CUSTOMER VALUES(
		'Nikki', 'Kaccaton', '723-543-1233', 'Nikki.Kaccaton@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'Brenda', 'Catnazaro', '723-543-2344', 'Brenda.Catnazaro@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'Bruce', 'LeCat', '723-543-3455', 'Bruce.LeCat@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'Betsy', 'Miller', '725-654-3211', 'Betsy.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'George', 'Miller', '725-654-4322', 'George.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'Kathy', 'Miller', '723-514-9877', 'Kathy.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
		'Betsy', 'Miller', '723-514-8766', 'Betsy.Miller@elsewhere.com');



/*****  INVOICE Data   ************************************************************/

INSERT INTO INVOICE VALUES(
		1, '04-Oct-15', '06-Oct-15', 158.50);
INSERT INTO INVOICE VALUES(
		2, '04-Oct-15', '06-Oct-15', 25.00);
INSERT INTO INVOICE VALUES(
		1, '06-Oct-15', '08-Oct-15', 49.00);
INSERT INTO INVOICE VALUES(
		4, '06-Oct-15', '08-Oct-15', 17.50);
INSERT INTO INVOICE VALUES(
		6, '07-Oct-15', '11-Oct-15', 12.00);
INSERT INTO INVOICE VALUES(
		3, '11-Oct-15', '13-Oct-15', 152.50);
INSERT INTO INVOICE VALUES(
		3, '11-Oct-15', '13-Oct-15', 7.00);
INSERT INTO INVOICE VALUES(
		7, '12-Oct-15', '14-Oct-15', 140.50);
INSERT INTO INVOICE VALUES(
		5, '12-Oct-15', '14-Oct-15', 27.00);


/*****  INVOICE_ITEM Data   ********************************************************/

INSERT INTO INVOICE_ITEM VALUES(2015001, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 2, 'Dress Shirt', 5,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 3, 'Formal Gown', 2, 10.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 4, 'Slacks–Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 5, 'Slacks–Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 6, 'Suit–Mens', 1,  9.00);
INSERT INTO INVOICE_ITEM VALUES(2015002, 1, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015003, 1, 'Slacks–Mens', 5,  5.00);
INSERT INTO INVOICE_ITEM VALUES(2015003, 2, 'Slacks–Womens', 4,  6.00);
INSERT INTO INVOICE_ITEM VALUES(2015004, 1, 'Dress Shirt', 7,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 2, 'Dress Shirt', 2,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 1, 'Blouse', 5,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 2, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 3, 'Slacks–Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015006, 4, 'Slacks–Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015007, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 1, 'Blouse', 3,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 2, 'Dress Shirt', 12, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 3, 'Slacks–Mens', 8,  5.00);
INSERT INTO INVOICE_ITEM VALUES(2015008, 4, 'Slacks–Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015009, 1, 'Suit–Mens', 3,  9.00);


/*********************************************************************************/


