/******************************************************************************/
/*									                                                        	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02    			*/
/*										                                                        */
/*	The Marcia's Dry Cleaing (MDC-CH02) Database - Create Tables              */
/*							                                                              */
/*	These are the Oracle Database 11g R2 and 12c SQL code solutions	                  */
/*								                                                            */
/******************************************************************************/

/******************************************************************************/

INSERT INTO CUSTOMER VALUES(
	1, 'Nikki', 'Kaccaton', '723-543-1233',
	'Nikki.Kaccaton@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	2, 'Brenda', 'Catnazaro', '723-543-2344',
	'Brenda.Catnazaro@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	3, 'Bruce', 'LeCat', '723-543-3455',
	'Bruce.LeCat@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	4, 'Betsy', 'Miller', '725-654-3211',
	'Betsy.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	5, 'George', 'Miller', '725-654-4322',
	'George.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	6, 'Kathy', 'Miller', '723-514-9877',
	'Kathy.Miller@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	7, 'Betsy', 'Miller', '723-514-8766',
	'Betsy.Miller@elsewhere.com');
  
COMMIT;

INSERT INTO INVOICE VALUES(
		2015001, 1,
		TO_DATE('10/04/15', 'MM/DD/YY'),
		TO_DATE('10/06/15', 'MM/DD/YY'),
		158.50);
INSERT INTO INVOICE VALUES(
		2015002, 2,
		TO_DATE('10/04/15', 'MM/DD/YY'),
		TO_DATE('10/06/15', 'MM/DD/YY'),
		25.00);
INSERT INTO INVOICE VALUES(
		2015003, 1,
		TO_DATE('10/06/15', 'MM/DD/YY'),
		TO_DATE('10/08/15', 'MM/DD/YY'),
		49.00);
INSERT INTO INVOICE VALUES(
		2015004, 4,
		TO_DATE('10/06/15', 'MM/DD/YY'),
		TO_DATE('10/08/15', 'MM/DD/YY'),
		17.50);
INSERT INTO INVOICE VALUES(
		2015005, 6,
		TO_DATE('10/07/15', 'MM/DD/YY'),
		TO_DATE('10/11/15', 'MM/DD/YY'),
		12.00);
INSERT INTO INVOICE VALUES(
		2015006, 3,
		TO_DATE('10/11/15', 'MM/DD/YY'),
		TO_DATE('10/13/15', 'MM/DD/YY'),
		152.50);
INSERT INTO INVOICE VALUES(
		2015007, 3,
		TO_DATE('10/11/15', 'MM/DD/YY'),
		TO_DATE('10/13/15', 'MM/DD/YY'),
		7.00);
INSERT INTO INVOICE VALUES(
		2015008, 7,
		TO_DATE('10/12/15', 'MM/DD/YY'),
		TO_DATE('10/14/15', 'MM/DD/YY'),
		140.50);
INSERT INTO INVOICE VALUES(
		2015009, 5,
		TO_DATE('10/12/15', 'MM/DD/YY'),
		TO_DATE('10/14/15', 'MM/DD/YY'),
		27.00);
    
COMMIT;

INSERT INTO INVOICE_ITEM VALUES(2015001, 1, 'Blouse', 2, 3.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 2, 'Dress Shirt', 5, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 3, 'Formal Gown', 2, 10.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 4, 'Slacks-Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 5, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 6, 'Suit-Mens', 1, 9.00);
INSERT INTO INVOICE_ITEM VALUES(2015002, 1, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015003, 1, 'Slacks-Mens', 5, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015003, 2, 'Slacks-Womens', 4, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015004, 1, 'Dress Shirt', 7, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 1, 'Blouse', 2, 3.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 2, 'Dress Shirt', 2, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 1, 'Blouse', 5, 3.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 2, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 3, 'Slacks-Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015006, 4, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015007, 1, 'Blouse', 2, 3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 1, 'Blouse', 3, 3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 2, 'Dress Shirt', 12, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 3, 'Slacks-Mens', 8, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015008, 4, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015009, 1, 'Suit-Mens', 3, 9.00);

COMMIT;











