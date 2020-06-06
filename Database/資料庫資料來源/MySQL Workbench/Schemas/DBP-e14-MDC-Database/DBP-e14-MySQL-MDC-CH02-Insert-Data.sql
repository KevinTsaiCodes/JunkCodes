/******************************************************************************/
/*					                                                        	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 2  			*/
/*										                                        */
/*	Marcia's Dry Cleaing (MDC) Database - Insert Data                           */
/*	These are the MySQL 5.6 SQL code solutions				                    */
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
/* */
  INSERT INTO INVOICE VALUES(
		2015001,1,'2015-10-04','2015-10-06',158.50);
INSERT INTO INVOICE VALUES(
		2015002,2,'2015-10-04','2015-10-06',25.00);
INSERT INTO INVOICE VALUES(
		2015003,1,'2015-10-06','2015-10-08',49.00);
INSERT INTO INVOICE VALUES(
		2015004,4,'2015-10-06','2015-10-08',17.50);
INSERT INTO INVOICE VALUES(
		2015005,6,'2015-10-07','2015-10-11',12.00);
INSERT INTO INVOICE VALUES(
		2015006,3,'2015-10-11','2015-10-13',152.50);
INSERT INTO INVOICE VALUES(
		2015007,3,'2015-10-11','2015-10-13',7.00);
INSERT INTO INVOICE VALUES(
		2015008,7,'2015-10-12','2015-10-14',140.50);
INSERT INTO INVOICE VALUES(
		2015009,5,'2015-10-12','2015-10-14',27.00); 
/* */
INSERT INTO INVOICE_ITEM VALUES(2015001, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 2, 'Dress Shirt', 5,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015001, 3, 'Formal Gown', 2, 10.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 4, 'Slacks-Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 5, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015001, 6, 'Suit-Mens', 1,  9.00);
INSERT INTO INVOICE_ITEM VALUES(2015002, 1, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015003, 1, 'Slacks-Mens', 5,  5.00);
INSERT INTO INVOICE_ITEM VALUES(2015003, 2, 'Slacks-Womens', 4,  6.00);
INSERT INTO INVOICE_ITEM VALUES(2015004, 1, 'Dress Shirt', 7,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015005, 2, 'Dress Shirt', 2,  2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 1, 'Blouse', 5,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 2, 'Dress Shirt', 10, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015006, 3, 'Slacks-Mens', 10, 5.00);
INSERT INTO INVOICE_ITEM VALUES(2015006, 4, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015007, 1, 'Blouse', 2,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 1, 'Blouse', 3,  3.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 2, 'Dress Shirt', 12, 2.50);
INSERT INTO INVOICE_ITEM VALUES(2015008, 3, 'Slacks-Mens', 8,  5.00);
INSERT INTO INVOICE_ITEM VALUES(2015008, 4, 'Slacks-Womens', 10, 6.00);
INSERT INTO INVOICE_ITEM VALUES(2015009, 1, 'Suit-Mens', 3,  9.00);
/* */