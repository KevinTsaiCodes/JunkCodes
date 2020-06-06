/******************************************************************************/
/*										                                    */
/*	Kroenke and Auer - Database Processing (14th Edition)             			*/
/*														                       */
/*	The Queen Anne Curiousity Shop Database Create Tables        				*/
/*														                        */
/*	These are the MySQL 5.6 code solutions	          */
/*														                       */
/******************************************************************************/

/*****   CUSTOMER DATA   ******************************************************/

INSERT INTO CUSTOMER VALUES (null, 
	'Shire', 'Robert', '6225 Evanston Ave N', 'Seattle', 'WA', '98103',
	'206-524-2433', 'Robert.Shire@somewhere.com');
  
INSERT INTO CUSTOMER VALUES (null, 
	'Goodyear', 'Katherine', '7335 11th Ave NE', 'Seattle', 'WA', '98105',
	'206-524-3544', 'Katherine.Goodyear@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Bancroft', 'Chris', '12605 NE 6th Street', 'Bellevue', 'WA', '98005',
	'425-635-9788', 'Chris.Bancroft@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Griffith', 'John', '335 Aloha Street', 'Seattle', 'WA', '98109',
	'206-524-4655', 'John.Griffith@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Tierney', 'Doris', '14510 NE 4th Street', 'Bellevue', 'WA', '98005',
	'425-635-8677', 'Doris.Tierney@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Anderson', 'Donna', '1410 Hillcrest Parkway', 'Mt. Vernon', 'WA', '98273',
	'360-538-7566', 'Donna.Anderson@elsewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Svane', 'Jack', '3211 42nd Street', 'Seattle', 'WA', '98115',
	'206-524-5766', 'Jack.Svane@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Walsh', 'Denesha', '6712 24th Avenue NE', 'Redmond', 'WA', '98053',
	'425-635-7566', 'Denesha.Walsh@somewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Enquist', 'Craig', '534 15th Street', 'Bellingham', 'WA', '98225',
	'360-538-6455', 'Craig.Enquist@elsewhere.com');
INSERT INTO CUSTOMER VALUES (null, 
	'Anderson', 'Rose', '6823 17th Ave NE', 'Seattle', 'WA', '98105',
	'206-524-6877', 'Rose.Anderson@elsewhere.com');
  
COMMIT;



/*****   ITEM DATA   **********************************************************/

/*****   ITEM Table As Used in Chapter02  ************************************/


INSERT INTO ITEM VALUES(null,
	'Antique Desk', 'European Specialties','2014-11-07', 1800.00, 3000.00);
INSERT INTO ITEM VALUES(null,
	'Antique Desk Chair', 'Andrew Lee','2014-11-10', 300.00, 500.00);
  
INSERT INTO ITEM VALUES(null,
	'Dining Table Linens', 'Linens and Things','2014-11-14', 600.00, 1000.00);
INSERT INTO ITEM VALUES(null,
	'Candles', 'Linens and Things','2014-11-14', 30.00, 50.00);
INSERT INTO ITEM VALUES(null,
	'Candles', 'Linens and Things','2014-11-14', 27.00, 45.00);
INSERT INTO ITEM VALUES(null,
	'Desk Lamp', 'Lamps and Lighting','2014-11-14', 150.00, 250.00);
INSERT INTO ITEM VALUES(null,
	'Dining Table Linens', 'Linens and Things','2014-11-14', 450.00, 750.00);
INSERT INTO ITEM VALUES(null,
	'Book Shelf', 'Denise Harrison','2014-11-21', 150.00, 250.00);
INSERT INTO ITEM VALUES(null,
	'Antique Chair', 'New York Brokerage','2014-11-21', 750.00, 1250.00);
INSERT INTO ITEM VALUES(null,
	'Antique Chair', 'New York Brokerage','2014-11-21', 1050.00, 1750.00);
INSERT INTO ITEM VALUES(null,
	'Antique Candle Holder', 'European Specialties','2014-11-28', 210.00, 350.00);
INSERT INTO ITEM VALUES(null,
	'Antique Desk', 'European Specialties','2015-01-05', 1920.00, 3200.00);
INSERT INTO ITEM VALUES(null,
	'Antique Desk', 'European Specialties','2015-01-05', 2100.00, 3500.00);
INSERT INTO ITEM VALUES(null,
	'Antique Desk Chair', 'Specialty Antiques','2015-01-06', 285.00, 475.00);
INSERT INTO ITEM VALUES(null,
	'Antique Desk Chair', 'Specialty Antiques','2015-01-06', 339.00, 565.00);
INSERT INTO ITEM VALUES(null,
	'Desk Lamp', 'General Antiques','2015-01-06', 150.00, 250.00);
INSERT INTO ITEM VALUES(null,
	'Desk Lamp', 'General Antiques','2015-01-06', 150.00, 250.00);
INSERT INTO ITEM VALUES(null,
	'Desk Lamp', 'Lamps and Lighting','2015-01-06', 144.00, 240.00);
INSERT INTO ITEM VALUES(null,
	'Antique Dining Table', 'Denesha Walsh','2015-01-10', 3000.00, 5000.00);
INSERT INTO ITEM VALUES(null,
	'Antique Sideboard', 'Chris Bancroft','2015-01-11', 2700.00, 4500.00);
INSERT INTO ITEM VALUES(null,
	'Dining Table Chairs', 'Specialty Antiques','2015-01-11', 5100.00, 8500.00);
INSERT INTO ITEM VALUES(null,
	'Dining Table Linens', 'Linens and Things','2015-01-12', 450.00, 750.00);
INSERT INTO ITEM VALUES(null,
	'Dining Table Linens', 'Linens and Things','2015-01-12', 480.00, 800.00);
INSERT INTO ITEM VALUES(null,
	'Candles', 'Linens and Things','2015-01-17', 30.00, 50.00);
INSERT INTO ITEM VALUES(null,
	'Candles', 'Linens and Things','2015-01-17', 36.00, 60.00);
COMMIT;  
  


/*****   SALE DATA   **********************************************************/

INSERT INTO SALE VALUES(null,
  1,'2014-12-14', 3500.00, 290.50, 3790.50);
  INSERT INTO SALE VALUES(null,
  2,'2014-12-15', 1000.00, 83.00, 1083.00);
INSERT INTO SALE VALUES(null,
  3, '2014-12-15', 50.00, 4.15, 54.15);
INSERT INTO SALE VALUES(null,
  4,'2014-12-23', 45.00, 3.74, 48.74);
INSERT INTO SALE VALUES(null,
  1,'2015-01-05', 250.00, 20.75, 270.75);
INSERT INTO SALE VALUES(null,
  5,'2015-01-10', 750.00, 62.25, 812.25);
INSERT INTO SALE VALUES(null,
  6,'2015-01-12', 250.00, 20.75, 270.75);
INSERT INTO SALE VALUES(null,
  2,'2015-01-15', 3000.00, 249.00, 3249.00);
INSERT INTO SALE VALUES(null,
  5,'2015-01-25', 350.00, 29.05, 379.05);
INSERT INTO SALE VALUES(null,
  7,'2015-02-04', 14250.00, 1182.75, 15432.75);
INSERT INTO SALE VALUES(null,
  8,'2015-02-04', 250.00, 20.75, 270.75);
INSERT INTO SALE VALUES(null,
  5,'2015-02-07', 50.00, 4.15, 54.15);
INSERT INTO SALE VALUES(null,
  9,'2015-02-07', 4500.00, 373.50, 4873.50);
INSERT INTO SALE VALUES(null,
  10,'2015-02-11', 3675.00, 305.03, 3980.03);
INSERT INTO SALE VALUES(null,
  2,'2015-02-11', 800.00, 66.40, 866.40);
  
COMMIT;

/*****   SALE_ITEM DATA   *****************************************************/

/*****   SALE_ITEM Table As Used in Chapter02  *******************************/

INSERT INTO SALE_ITEM VALUES(1, 1, 1, 3000.00);
INSERT INTO SALE_ITEM VALUES(1, 2, 2, 500.00);
INSERT INTO SALE_ITEM VALUES(2,1,3, 1000.00);
INSERT INTO SALE_ITEM VALUES(3,1,4, 50.00);
INSERT INTO SALE_ITEM VALUES(4,1,5, 45.00);
INSERT INTO SALE_ITEM VALUES(5,1,6, 250.00);
INSERT INTO SALE_ITEM VALUES(6,1,7, 750.00);
INSERT INTO SALE_ITEM VALUES(7,1,8, 250.00);
INSERT INTO SALE_ITEM VALUES(8,1,9, 1250.00);
INSERT INTO SALE_ITEM VALUES(8,2,10, 1750.00);
INSERT INTO SALE_ITEM VALUES(9,1,11, 350.00);
INSERT INTO SALE_ITEM VALUES(10,1,19, 5000.00);
INSERT INTO SALE_ITEM VALUES(10, 2, 21, 8500.00);
INSERT INTO SALE_ITEM VALUES(10, 3, 22, 750.00);
INSERT INTO SALE_ITEM VALUES(11, 1, 17, 250.00);
INSERT INTO SALE_ITEM VALUES(12, 1, 24, 50.00);
INSERT INTO SALE_ITEM VALUES(13, 1, 20, 4500.00);
INSERT INTO SALE_ITEM VALUES(14, 1, 12, 3200.00);
INSERT INTO SALE_ITEM VALUES(14, 2, 14, 475.00);
INSERT INTO SALE_ITEM VALUES(15, 1, 23, 800.00);

COMMIT;
/********************************************************************************/