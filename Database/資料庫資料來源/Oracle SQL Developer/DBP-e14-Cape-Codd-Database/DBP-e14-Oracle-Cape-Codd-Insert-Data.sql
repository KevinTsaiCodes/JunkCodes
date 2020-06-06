/******************************************************************************/
/*																				                                    */
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02			    */
/*																				                                    */
/*	Cape Codd Outdoor Sports - Insert Data										                */
/*																				                                    */
/*	These are the ORACLE Database SQL code solutions				                  */
/*																				                                    */
/******************************************************************************/

INSERT INTO RETAIL_ORDER VALUES (
	1000, 10, '98110', 'December', 2014, 445);
INSERT INTO RETAIL_ORDER VALUES (
	2000, 20, '02335', 'December', 2014, 310);
INSERT INTO RETAIL_ORDER VALUES (
	3000, 10, '98110', 'January', 2015, 480);


INSERT INTO SKU_DATA VALUES (
	100100, 'Std. Scuba Tank, Yellow', 'Water Sports', 'Pete Hansen');
INSERT INTO SKU_DATA VALUES (
	100200, 'Std. Scuba Tank, Magenta', 'Water Sports', 'Pete Hansen');
INSERT INTO SKU_DATA VALUES (
	101100, 'Dive Mask, Small Clear', 'Water Sports', 'Nancy Meyers');
INSERT INTO SKU_DATA VALUES (
	101200, 'Dive Mask, Med Clear', 'Water Sports', 'Nancy Meyers');
INSERT INTO SKU_DATA VALUES (
	201000, 'Half-dome Tent', 'Camping', 'Cindy Lo');		
INSERT INTO SKU_DATA VALUES (
	202000, 'Half-dome Tent Vestibule', 'Camping', 'Cindy Lo');
INSERT INTO SKU_DATA VALUES (
	301000, 'Light Fly Climbing Harness', 'Climbing', 'Jerry Martin');		
INSERT INTO SKU_DATA VALUES (
	302000, 'Locking Carabiner, Oval', 'Climbing', 'Jerry Martin');	


INSERT INTO ORDER_ITEM VALUES (
	1000, 201000, 1, 300, 300);
INSERT INTO ORDER_ITEM VALUES (
	1000, 202000, 1, 130, 130);
INSERT INTO ORDER_ITEM VALUES (
	2000, 101100, 4, 50, 200);
INSERT INTO ORDER_ITEM VALUES (
	2000, 101200, 2, 50, 100);
INSERT INTO ORDER_ITEM VALUES (
	3000, 100200, 1, 300, 300);
INSERT INTO ORDER_ITEM VALUES (
	3000, 101100, 2, 50, 100);
INSERT INTO ORDER_ITEM VALUES (
	3000, 101200, 1, 50, 50);

/*  SQL code to insert the data into WAREHOUSE:  */
						
INSERT INTO WAREHOUSE VALUES (
	100, 'Atlanta', 'GA', 'Dave Jones', 125000);
INSERT INTO WAREHOUSE VALUES (
	200, 'Chicago', 'IL', 'Lucille Smith', 100000);
INSERT INTO WAREHOUSE VALUES (
	300, 'Bangor', 'ME', 'Bart Evans', 150000);
INSERT INTO WAREHOUSE VALUES (
	400, 'Seattle', 'WA', 'Dale Rogers', 130000);
INSERT INTO WAREHOUSE VALUES (
	500, 'San Francisco', 'CA','Grace Jefferson', 200000);

	
/*  SQL code to insert the data into INVENTORY:   */

INSERT INTO INVENTORY VALUES (
	100, 100100, 'Std. Scuba Tank, Yellow', 250, 0);
INSERT INTO INVENTORY VALUES (
	200, 100100, 'Std. Scuba Tank, Yellow', 100, 50);
INSERT INTO INVENTORY VALUES (
	300, 100100, 'Std. Scuba Tank, Yellow', 100, 0);
INSERT INTO INVENTORY VALUES (
	400, 100100, 'Std. Scuba Tank, Yellow', 200, 0);
INSERT INTO INVENTORY VALUES (
	100, 100200, 'Std. Scuba Tank, Magenta', 200, 30);
INSERT INTO INVENTORY VALUES (
	200, 100200, 'Std. Scuba Tank, Magenta', 75, 75);
INSERT INTO INVENTORY VALUES (
	300, 100200, 'Std. Scuba Tank, Magenta', 100, 100);
INSERT INTO INVENTORY VALUES (
	400, 100200, 'Std. Scuba Tank, Magenta', 250, 0);
INSERT INTO INVENTORY VALUES (
	100, 101100, 'Dive Mask, Small Clear', 0, 500);
INSERT INTO INVENTORY VALUES (
	200, 101100, 'Dive Mask, Small Clear', 0, 500);
INSERT INTO INVENTORY VALUES (
	300, 101100, 'Dive Mask, Small Clear', 300, 200);
INSERT INTO INVENTORY VALUES (
	400, 101100, 'Dive Mask, Small Clear', 450, 0);
INSERT INTO INVENTORY VALUES (
	100, 101200, 'Dive Mask, Med Clear', 100, 500);
INSERT INTO INVENTORY VALUES (
	200, 101200, 'Dive Mask, Med Clear', 50, 500);
INSERT INTO INVENTORY VALUES (
	300, 101200, 'Dive Mask, Med Clear', 475, 0);
INSERT INTO INVENTORY VALUES (
	400, 101200, 'Dive Mask, Med Clear', 250, 250);
INSERT INTO INVENTORY VALUES (
	100, 201000, 'Half-dome Tent', 2, 100);
INSERT INTO INVENTORY VALUES (
	200, 201000, 'Half-dome Tent', 10, 250);
INSERT INTO INVENTORY VALUES (
	300, 201000, 'Half-dome Tent', 250, 0);
INSERT INTO INVENTORY VALUES (
	400, 201000, 'Half-dome Tent', 0, 250);
INSERT INTO INVENTORY VALUES (
	100, 202000, 'Half-dome Tent Vestibule', 10, 250);
INSERT INTO INVENTORY VALUES (
	200, 202000, 'Half-dome Tent Vestibule', 1, 250);
INSERT INTO INVENTORY VALUES (
	300, 202000, 'Half-dome Tent Vestibule', 100, 0);
INSERT INTO INVENTORY VALUES (
	400, 202000, 'Half-dome Tent Vestibule', 0, 200);
INSERT INTO INVENTORY VALUES (
	100, 301000, 'Light Fly Climbing Harness', 300, 250);
INSERT INTO INVENTORY VALUES (
	200, 301000, 'Light Fly Climbing Harness', 250, 250);
INSERT INTO INVENTORY VALUES (
	300, 301000, 'Light Fly Climbing Harness', 0, 250);
INSERT INTO INVENTORY VALUES (
	400, 301000, 'Light Fly Climbing Harness', 0, 250);
INSERT INTO INVENTORY VALUES (
	100, 302000, 'Locking Carabiner, Oval', 1000, 0);
INSERT INTO INVENTORY VALUES (
	200, 302000, 'Locking Carabiner, Oval', 1250, 0);
INSERT INTO INVENTORY VALUES (
	300, 302000, 'Locking Carabiner, Oval', 500, 500);
INSERT INTO INVENTORY VALUES (
	400, 302000, 'Locking Carabiner, Oval', 0, 1000);

/*  SQL code to insert the data into CATALOG_SKU_2013:   */

INSERT INTO CATALOG_SKU_2013 VALUES (
	 seqCSKU13.NextVal, 100100, 'Std. Scuba Tank, Yellow', 'Water Sports', 23,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES ( 
	seqCSKU13.NextVal, 100500, 'Std. Scuba Tank, Light Green', 'Water Sports', NULL,
  TO_DATE('07/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES ( 
	 seqCSKU13.NextVal, 100600, 'Std. Scuba Tank, Dark Green', 'Water Sports', NULL,
   TO_DATE('07/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES (    
	 seqCSKU13.NextVal, 101100, 'Dive Mask, Small Clear', 'Water Sports', 24,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES ( 
	 seqCSKU13.NextVal, 101200, 'Dive Mask, Med Clear', 'Water Sports', 24,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES (    
   seqCSKU13.NextVal, 201000, 'Half-dome Tent', 'Camping', 45,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES (
   seqCSKU13.NextVal, 202000, 'Half-dome Tent Vestibule', 'Camping', 47,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES (    
	 seqCSKU13.NextVal, 301000, 'Light Fly Climbing Harness', 'Climbing', 76,
   TO_DATE('01/01/13', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2013 VALUES (    
	 seqCSKU13.NextVal, 302000, 'Locking Carabiner, Oval', 'Climbing', 78,
   TO_DATE('01/01/13', 'MM/DD/YY'));
  
/*  SQL code to insert the data into CATALOG_SKU_2014:   */
  
INSERT INTO CATALOG_SKU_2014 VALUES (
	seqCSKU14.NextVal, 100100, 'Std. Scuba Tank, Yellow', 'Water Sports', 23,
  TO_DATE('01/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES (
  seqCSKU14.NextVal, 100300, 'Std. Scuba Tank, Light Blue', 'Water Sports', 23,
  TO_DATE('01/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 100400, 'Std. Scuba Tank, Dark Blue', 'Water Sports', NULL,
  TO_DATE('08/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 101100, 'Dive Mask, Small Clear', 'Water Sports', 26,
  TO_DATE('01/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 101200, 'Dive Mask, Med Clear', 'Water Sports', 26,
  TO_DATE('01/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 201000, 'Half-dome Tent', 'Camping', 46,
  TO_DATE('01/01/14', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 202000, 'Half-dome Tent Vestibule', 'Camping', 46,
  TO_DATE('01/01/14', 'MM/DD/YY'));
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 301000, 'Light Fly Climbing Harness', 'Climbing', 77,
  TO_DATE('01/01/14', 'MM/DD/YY'));		
INSERT INTO CATALOG_SKU_2014 VALUES	(
  seqCSKU14.NextVal, 302000, 'Locking Carabiner, Oval', 'Climbing', 79,
  TO_DATE('01/01/14', 'MM/DD/YY'));	 
  
/*  SQL code to insert the data into CATALOG_SKU_2015:   */
  
INSERT INTO CATALOG_SKU_2015 VALUES (
	 seqCSKU15.NextVal,100100, 'Std. Scuba Tank, Yellow', 'Water Sports', 23,
   TO_DATE('01/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,100200, 'Std. Scuba Tank, Magenta', 'Water Sports', 23,
  TO_DATE('01/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,101100, 'Dive Mask, Small Clear', 'Water Sports', 27,
  TO_DATE('08/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,101200, 'Dive Mask, Med Clear', 'Water Sports', 27,
  TO_DATE('01/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,201000, 'Half-dome Tent', 'Camping', 45,
  TO_DATE('01/01/15', 'MM/DD/YY'));		
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,202000, 'Half-dome Tent Vestibule', 'Camping', 45,
  TO_DATE('01/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,203000, 'Half-dome Tent Vestibule - Wide', 'Camping', NULL,
  TO_DATE('04/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,301000, 'Light Fly Climbing Harness', 'Climbing', 76,
  TO_DATE('01/01/15', 'MM/DD/YY'));	
INSERT INTO CATALOG_SKU_2015 VALUES	(
  seqCSKU15.NextVal,302000, 'Locking Carabiner, Oval', 'Climbing', 78,
  TO_DATE('01/01/15', 'MM/DD/YY'));	
  
COMMIT;
