/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			*/
/*																				*/
/*	Morgan Importing [MI-CH02] Project Insert Data								*/
/*																				*/
/*	Microsoft SQL Server 2012 / 2014 ALcode solutions							*/
/*																				*/
/********************************************************************************/

/*****   ITEM Data   ***********************************************************/

INSERT INTO ITEM VALUES
		('QE Dining Set', '07-Apr-15', 'Eastern Treasures',
		'Manila', 2, 403405, 0.01774);
INSERT INTO ITEM
		VALUES('Willow Serving Dishes', '15-Jul-15',  'Jade Antiques',
		'Singapore', 75, 102, 0.5903);
INSERT INTO ITEM
		VALUES('Large Bureau', '17-Jul-15', 'Eastern Sales',
		'Singapore', 8, 2000, 0.5903);
INSERT INTO ITEM
		VALUES('Brass Lamps', '20-Jul-15',  'Jade Antiques',
		'Singapore', 40, 50, 0.5903);

/*****   SHIPMENT Data   ***********************************************************/

INSERT INTO SHIPMENT VALUES
	('ABC Trans-Oceanic', 2008651, '10-Dec-14', '15-Mar-15', 15000.00);
INSERT INTO SHIPMENT VALUES
	('ABC Trans-Oceanic', 2009012, '10-Jan-15', '20-Mar-15', 12000.00);
INSERT INTO SHIPMENT VALUES
	('Worldwide', 49100300, '05-May-15', '17-Jun-15', 20000.00);
INSERT INTO SHIPMENT VALUES
	('International', 399400, '02-Jun-15', '17-Jul-15', 17500.00 );
INSERT INTO SHIPMENT VALUES
	('Worldwide', 84899440, '10-Jul-15', '28-Jul-15', 25000.00);
INSERT INTO SHIPMENT VALUES
	('International', 488955,  '05-Aug-15', '11-Sep-15', 18000.00);

/*****   SHIPMENT Item   ***********************************************************/

INSERT INTO SHIPMENT_ITEM VALUES(3, 1, 1, 15000);
INSERT INTO SHIPMENT_ITEM VALUES(4, 1, 4, 1200);
INSERT INTO SHIPMENT_ITEM VALUES(4, 2, 3, 9500);
INSERT INTO SHIPMENT_ITEM VALUES(4, 3, 2, 4500);


/********************************************************************************/
