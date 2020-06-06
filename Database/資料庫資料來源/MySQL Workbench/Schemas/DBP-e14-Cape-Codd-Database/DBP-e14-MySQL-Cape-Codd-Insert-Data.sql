/********************************************************************************/
/*																				*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02/10C		*/
/*																				*/
/*	Cape Codd Outdoor Sports Database - Insert Data 							*/
/*																				*/
/*	These are the MySQL 5.6 SQL code solutions				                    */
/*																				*/
/********************************************************************************/

USE cape_codd;

/***   RETAIL_ORDER data   ******************************************************/

INSERT INTO RETAIL_ORDER VALUES (
	1000, 10, '98110', 'December', 2014, 445);
INSERT INTO RETAIL_ORDER VALUES (
	2000, 20, '02335', 'December', 2014, 310);
INSERT INTO RETAIL_ORDER VALUES (
	3000, 10, '98110', 'January', 2015, 480);

/***   SKU_DATA data   **********************************************************/

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

/***   ORDER_ITEM data   ********************************************************/

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

/***   WAREHOUSE data   *********************************************************/

INSERT INTO WAREHOUSE VALUES (
	100, 'Atlanta', 'GA','Dave Jones', 125000);
INSERT INTO WAREHOUSE VALUES (
	200, 'Chicago', 'IL', 'Lucille Smith', 100000);
INSERT INTO WAREHOUSE VALUES (
	300, 'Bangor', 'ME', 'Bart Evans', 150000);
INSERT INTO WAREHOUSE VALUES (
	400, 'Seattle', 'WA','Dale Rogers', 130000);	
INSERT INTO WAREHOUSE VALUES (
	500, 'San Francisco', 'CA','Grace Jefferson', 200000);	    


/***   INVENTORY data   *********************************************************/

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


/********************************************************************************/
/* catalog_sku_2013 */
insert into catalog_sku_2013 values(null, 100100, "Std. Scuba Tank, Yellow", "Water Sports", 23, "2013-01-01");
insert into catalog_sku_2013 values(null, 100500, "Std. Scuba Tank, Light Green", "Water Sports", null, "2013-07-01");
insert into catalog_sku_2013 values(null, 100600, "Std. Scuba Tank, Dark Green", "Water Sports", null, "2013-07-01");
insert into catalog_sku_2013 values(null, 101100, "Dive Mask, Small Clear", "Water Sports", 24, "2013-01-01");
insert into catalog_sku_2013 values(null, 101200, "Dive Mask, Med Clear", "Water Sports", 24, "2013-01-01");
insert into catalog_sku_2013 values(null, 201000, "Half-dome Tent", "Camping", 45, "2013-01-01");
insert into catalog_sku_2013 values(null, 202000, "Half-dome Tent Vestibule", "Camping", 47, "2013-01-01");
insert into catalog_sku_2013 values(null, 301000, "Light Fly Climbing Harness", "Climbing", 76, "2013-01-01");
insert into catalog_sku_2013 values(null, 302000, "Locking Carabiner, Oval", "Climbing", 76, "2013-01-01");

/* catalog_sku_2014 */
insert into catalog_sku_2014 values(null, 100100, "Std. Scuba Tank, Yellow", "Water Sports", 23, "2014-01-01");
insert into catalog_sku_2014 values(null, 100300, "Std. Scuba Tank, Light Blue", "Water Sports", 23, "2014-01-01");
insert into catalog_sku_2014 values(null, 100400, "Std. Scuba Tank, Dark Blue", "Water Sports", null, "2014-08-01");
insert into catalog_sku_2014 values(null, 101100, "Dive Mask, Small Clear", "Water Sports", 26, "2014-01-01");
insert into catalog_sku_2014 values(null, 101200, "Dive Mask, Med Clear", "Water Sports", 26, "2014-01-01");
insert into catalog_sku_2014 values(null, 201000, "Half-dome Tent", "Camping", 46, "2014-01-01");
insert into catalog_sku_2014 values(null, 202000, "Half-dome Tent Vestibule", "Camping", 46, "2014-01-01");
insert into catalog_sku_2014 values(null, 301000, "Light Fly Climbing Harness", "Climbing", 77, "2014-01-01");
insert into catalog_sku_2014 values(null, 302000, "Locking Carabiner, Oval", "Climbing", 79, "2014-01-01");

/*  SQL code to insert the data into CATALOG_SKU_2015:   */
  
INSERT INTO CATALOG_SKU_2015 VALUES (
	null,100100, 'Std. Scuba Tank, Yellow', 'Water Sports', 23,"2015-01-01");
INSERT INTO CATALOG_SKU_2015 VALUES	(
	null,100200, 'Std. Scuba Tank, Magenta', 'Water Sports', 23,"2015-01-01");
INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,101100, 'Dive Mask, Small Clear', 'Water Sports', 27,"2015-08-01");
 INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,101200, 'Dive Mask, Med Clear', 'Water Sports', 27,"2015-01-01");
 INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,201000, 'Half-dome Tent', 'Camping', 45,"2015-01-01");
 INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,202000, 'Half-dome Tent Vestibule', 'Camping', 45,"2015-01-01");
 INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,203000, 'Half-dome Tent Vestibule - Wide', 'Camping', NULL,"2015-04-01");
  INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,301000, 'Light Fly Climbing Harness', 'Climbing', 76,"2015-01-01");
 INSERT INTO CATALOG_SKU_2015 VALUES	(
  null,302000, 'Locking Carabiner, Oval', 'Climbing', 78,"2015-01-01");

  




