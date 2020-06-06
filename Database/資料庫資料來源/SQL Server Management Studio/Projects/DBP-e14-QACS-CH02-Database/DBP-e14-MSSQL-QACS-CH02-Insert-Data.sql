/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			*/
/*																				*/
/*	The Queen Anne Curiousity Shop [QACS] Database Data							*/
/*																				*/
/*	These are the Microsoft SQL Server 2012 /2014 SQL code solutions			*/
/*																				*/
/********************************************************************************/

/*****   CUSTOMER DATA   ********************************************************/

INSERT INTO CUSTOMER VALUES(
	'Shire', 'Robert', '6225 Evanston Ave N', 'Seattle', 'WA', '98103',
	'206-524-2433', 'Robert.Shire@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Goodyear', 'Katherine', '7335 11th Ave NE', 'Seattle', 'WA', '98105',
	'206-524-3544', 'Katherine.Goodyear@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Bancroft', 'Chris', '12605 NE 6th Street', 'Bellevue', 'WA', '98005',
	'425-635-9788', 'Chris.Bancroft@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Griffith', 'John', '335 Aloha Street', 'Seattle', 'WA', '98109',
	'206-524-4655', 'John.Griffith@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Tierney', 'Doris', '14510 NE 4th Street', 'Bellevue', 'WA', '98005',
	'425-635-8677', 'Doris.Tierney@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Anderson', 'Donna', '1410 Hillcrest Parkway', 'Mt. Vernon', 'WA', '98273',
	'360-538-7566', 'Donna.Anderson@elsewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Svane', 'Jack', '3211 42nd Street', 'Seattle', 'WA', '98115',
	'206-524-5766', 'Jack.Svane@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Walsh', 'Denesha', '6712 24th Avenue NE', 'Redmond', 'WA', '98053',
	'425-635-7566', 'Denesha.Walsh@somewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Enquist', 'Craig', '534 15th Street', 'Bellingham', 'WA', '98225',
	'360-538-6455', 'Craig.Enquist@elsewhere.com');
INSERT INTO CUSTOMER VALUES(
	'Anderson', 'Rose', '6823 17th Ave NE', 'Seattle', 'WA', '98105',
	'206-524-6877', 'Rose.Anderson@elsewhere.com');


/*****   ITEM DATA   ************************************************************/

/*****   ITEM Table As Used in Chapter02   **************************************/

INSERT INTO ITEM VALUES(
	'Antique Desk', 'European Specialties', '07-Nov-14', 1800.00, 3000.00);
INSERT INTO ITEM VALUES(
	'Antique Desk Chair', 'Andrew Lee', '10-Nov-14', 300.00, 500.00);
INSERT INTO ITEM VALUES(
	'Dining Table Linens', 'Linens and Things', '14-Nov-14', 600.00, 1000.00);
INSERT INTO ITEM VALUES(
	'Candles', 'Linens and Things', '14-Nov-14', 30.00, 50.00);
INSERT INTO ITEM VALUES(
	'Candles', 'Linens and Things', '14-Nov-14', 27.00, 45.00);
INSERT INTO ITEM VALUES(
	'Desk Lamp', 'Lamps and Lighting', '14-Nov-14', 150.00, 250.00);
INSERT INTO ITEM VALUES(
	'Dining Table Linens', 'Linens and Things', '14-Nov-14', 450.00, 750.00);
INSERT INTO ITEM VALUES(
	'Book Shelf', 'Denise Harrison', '21-Nov-14', 150.00, 250.00);
INSERT INTO ITEM VALUES(
	'Antique Chair', 'New York Brokerage', '21-Nov-14', 750.00, 1250.00);
INSERT INTO ITEM VALUES(
	'Antique Chair', 'New York Brokerage', '21-Nov-14', 1050.00, 1750.00);
INSERT INTO ITEM VALUES(
	'Antique Candle Holder', 'European Specialties', '28-Nov-14', 210.00, 350.00);
INSERT INTO ITEM VALUES(
	'Antique Desk', 'European Specialties', '05-Jan-15', 1920.00, 3200.00);
INSERT INTO ITEM VALUES(
	'Antique Desk', 'European Specialties', '05-Jan-15', 2100.00, 3500.00);
INSERT INTO ITEM VALUES(
	'Antique Desk Chair', 'Specialty Antiques', '06-Jan-15', 285.00, 475.00);
INSERT INTO ITEM VALUES(
	'Antique Desk Chair', 'Specialty Antiques', '06-Jan-15', 339.00, 565.00);
INSERT INTO ITEM VALUES(
	'Desk Lamp', 'General Antiques', '06-Jan-15', 150.00, 250.00);
INSERT INTO ITEM VALUES(
	'Desk Lamp', 'General Antiques', '06-Jan-15', 150.00, 250.00);
INSERT INTO ITEM VALUES(
	'Desk Lamp', 'Lamps and Lighting', '06-Jan-15', 144.00, 240.00);
INSERT INTO ITEM VALUES(
	'Antique Dining Table', 'Denesha Walsh', '10-Jan-15', 3000.00, 5000.00);
INSERT INTO ITEM VALUES(
	'Antique Sideboard', 'Chris Bancroft', '11-Jan-15', 2700.00, 4500.00);
INSERT INTO ITEM VALUES(
	'Dining Table Chairs', 'Specialty Antiques', '11-Jan-15', 5100.00, 8500.00);
INSERT INTO ITEM VALUES(
	'Dining Table Linens', 'Linens and Things', '12-Jan-15', 450.00, 750.00);
INSERT INTO ITEM VALUES(
	'Dining Table Linens', 'Linens and Things', '12-Jan-15', 480.00, 800.00);
INSERT INTO ITEM VALUES(
	'Candles', 'Linens and Things', '17-Jan-15', 30.00, 50.00);
INSERT INTO ITEM VALUES(
	'Candles', 'Linens and Things', '17-Jan-15', 36.00, 60.00);

/*****   SALE DATA   ************************************************************/

INSERT INTO SALE VALUES(1, '14-Dec-14', 3500.00, 290.50, 3790.50); 
INSERT INTO SALE VALUES(2, '15-Dec-14', 1000.00, 83.00, 1083.00); 
INSERT INTO SALE VALUES(3, '15-Dec-14', 50.00, 4.15, 54.15); 
INSERT INTO SALE VALUES(4, '23-Dec-14', 45.00, 3.74, 48.74); 
INSERT INTO SALE VALUES(1, '05-Jan-15', 250.00, 20.75, 270.75); 
INSERT INTO SALE VALUES(5, '10-Jan-15', 750.00, 62.25, 812.25);
INSERT INTO SALE VALUES(6, '12-Jan-15', 250.00, 20.75, 270.75); 
INSERT INTO SALE VALUES(2, '15-Jan-15', 3000.00, 249.00, 3249.00);
INSERT INTO SALE VALUES(5, '25-Jan-15', 350.00, 29.05, 379.05);
INSERT INTO SALE VALUES(7, '04-Feb-15', 14250.00, 1182.75, 15432.75);
INSERT INTO SALE VALUES(8, '04-Feb-15', 250.00, 20.75, 270.75);
INSERT INTO SALE VALUES(5, '07-Feb-15', 50.00, 4.15, 54.15);
INSERT INTO SALE VALUES(9, '07-Feb-15', 4500.00, 373.50, 4873.50);
INSERT INTO SALE VALUES(10, '11-Feb-15', 3675.00, 305.03, 3980.03); 
INSERT INTO SALE VALUES(2, '11-Feb-15', 800.00, 66.40, 866.40);


/*****   SALE_ITEM DATA   *******************************************************/

/*****   SALE_ITEM Table As Used in Chapter 02   ********************************/

INSERT INTO SALE_ITEM VALUES(1,	1, 1, 3000.00); 
INSERT INTO SALE_ITEM VALUES(1,	2, 2, 500.00); 
INSERT INTO SALE_ITEM VALUES(2,	1, 3, 1000.00); 
INSERT INTO SALE_ITEM VALUES(3,	1, 4, 50.00);
INSERT INTO SALE_ITEM VALUES(4,	1, 5, 45.00); 
INSERT INTO SALE_ITEM VALUES(5,	1, 6, 250.00); 
INSERT INTO SALE_ITEM VALUES(6,	1, 7, 750.00); 
INSERT INTO SALE_ITEM VALUES(7,	1, 8, 250.00); 
INSERT INTO SALE_ITEM VALUES(8,	1, 9, 1250.00); 
INSERT INTO SALE_ITEM VALUES(8,	2, 10, 1750.00); 
INSERT INTO SALE_ITEM VALUES(9,	1, 11, 350.00); 
INSERT INTO SALE_ITEM VALUES(10, 1, 19, 5000.00); 
INSERT INTO SALE_ITEM VALUES(10, 2, 21, 8500.00); 
INSERT INTO SALE_ITEM VALUES(10, 3, 22, 750.00); 
INSERT INTO SALE_ITEM VALUES(11, 1, 17, 250.00); 
INSERT INTO SALE_ITEM VALUES(12, 1, 24, 50.00); 
INSERT INTO SALE_ITEM VALUES(13, 1, 20, 4500.00); 
INSERT INTO SALE_ITEM VALUES(14, 1, 12, 3200.00); 
INSERT INTO SALE_ITEM VALUES(14, 2, 14, 475.00); 
INSERT INTO SALE_ITEM VALUES(15, 1, 23, 800.00); 

/********************************************************************************/
