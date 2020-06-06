/******************************************************************************/
/*									                                       	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02		  */
/*										                                      */
/*	The Morgan Importing (MI-CH02) Database - Insert Data                     */
/*							                                                 */
/*	These are the MySQL 5.6 code solutions							          */
/*								                                               */
/******************************************************************************/

 
/*****   ITEM Data   *************************************************/

INSERT INTO ITEM VALUES(
		null, 'QE Dining set','2015-04-07', 
		 'Eastern Treasures', 'Manila', 2, 403405, 0.01774);
INSERT INTO ITEM VALUES(
		null, 'Willow Serving Dishes','2015-07-15', 
		 'Jade Antiques', 'Singapore', 75, 102, 0.5903);
INSERT INTO ITEM VALUES(
		null, 'Large Bureau','2015-07-17', 
		 'Eastern Sales', 'Singapore', 8, 2000, 0.5903);
INSERT INTO ITEM VALUES(
		null, 'Brass Lamps','2015-07-20', 
		 'Jade Antiques', 'Singapore', 40, 50, 0.5903);

COMMIT;
 

/*****   SHIPMENT Data   ***************************************************/

INSERT INTO SHIPMENT VALUES(
		null, 'ABC Trans-Oceanic', 2008651,'2014-12-10','2015-03-15', 15000.00);
INSERT INTO SHIPMENT VALUES(
		null, 'ABC Trans-Oceanic', 2009012,'2015-01-10','2015-03-20', 12000.00);
INSERT INTO SHIPMENT VALUES(
		null, 'Worldwide', 49100300,'2015-05-05','2015-06-17', 20000.00);
INSERT INTO SHIPMENT VALUES(
		null, 'International', 399400,'2015-06-02','2015-07-17', 17500.00 );
INSERT INTO SHIPMENT VALUES(
		null, 'Worldwide', 84899440,'2015-07-10','2015-07-28', 25000.00);
INSERT INTO SHIPMENT VALUES(
		null, 'International', 488955,'2015-08-05','2015-09-11', 18000.00);

COMMIT;

/*****   SHIPMENT_ITEM Data   **********************************************/

INSERT INTO SHIPMENT_ITEM VALUES(3, 1, 1, 15000);
INSERT INTO SHIPMENT_ITEM VALUES(4, 1, 4, 1200);
INSERT INTO SHIPMENT_ITEM VALUES(4, 2, 3, 9500);
INSERT INTO SHIPMENT_ITEM VALUES(4, 3, 2, 4500);

COMMIT;
