/******************************************************************************/
/*									                                                        	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02   			  */
/*										                                                        */
/*	The Morgan Importing (MI-CH02) Database - Insert Data                     */
/*							                                                              */
/*	These are the Oracle Database 12c and 11g Release 2 SQL code solutions	          */
/*								                                                            */
/******************************************************************************/

 
/*****   ITEM Data   *************************************************/

INSERT INTO ITEM VALUES(
		seqIID.NextVal, 'QE Dining set', TO_DATE('07-Apr-15', 'DD-Mon-YY'), 
		 'Eastern Treasures', 'Manila', 2, 403405, 0.01774);
INSERT INTO ITEM VALUES(
		seqIID.NextVal, 'Willow Serving Dishes', TO_DATE('15-Jul-15', 'DD-Mon-YY'), 
		 'Jade Antiques', 'Singapore', 75, 102, 0.5903);
INSERT INTO ITEM VALUES(
		seqIID.NextVal, 'Large Bureau', TO_DATE('17-Jul-15', 'DD-Mon-YY'), 
		 'Eastern Sales', 'Singapore', 8, 2000, 0.5903);
INSERT INTO ITEM VALUES(
		seqIID.NextVal, 'Brass Lamps', TO_DATE('20-Jul-15', 'DD-Mon-YY'), 
		 'Jade Antiques', 'Singapore', 40, 50, 0.5903);

COMMIT;
 

/*****   SHIPMENT Data   ***************************************************/

INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'ABC Trans-Oceanic', 2008651, TO_DATE('10-Dec-14' , 'DD-Mon-YY'), TO_DATE('15-Mar-15', 'DD-Mon-YY'), 15000.00);
INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'ABC Trans-Oceanic', 2009012, TO_DATE('10-Jan-15', 'DD-Mon-YY'), TO_DATE('20-Mar-15', 'DD-Mon-YY'), 12000.00);
INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'Worldwide', 49100300, TO_DATE('05-May-15', 'DD-Mon-YY'), TO_DATE('17-Jun-15', 'DD-Mon-YY'), 20000.00);
INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'International', 399400, TO_DATE('02-Jun-15', 'DD-Mon-YY'), TO_DATE('17-Jul-15', 'DD-Mon-YY'), 17500.00 );
INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'Worldwide', 84899440, TO_DATE('10-Jul-15', 'DD-Mon-YY'), TO_DATE('28-Jul-15', 'DD-Mon-YY'), 25000.00);
INSERT INTO SHIPMENT VALUES(
		seqShipmentID.NextVal, 'International', 488955, TO_DATE('05-Aug-15', 'DD-Mon-YY'), TO_DATE('11-Sep-15', 'DD-Mon-YY'), 18000.00);

COMMIT;

/*****   SHIPMENT_ITEM Data   **********************************************/

INSERT INTO SHIPMENT_ITEM VALUES(3, 1, 1, 15000);
INSERT INTO SHIPMENT_ITEM VALUES(4, 1, 4, 1200);
INSERT INTO SHIPMENT_ITEM VALUES(4, 2, 3, 9500);
INSERT INTO SHIPMENT_ITEM VALUES(4, 3, 2, 4500);

COMMIT;
