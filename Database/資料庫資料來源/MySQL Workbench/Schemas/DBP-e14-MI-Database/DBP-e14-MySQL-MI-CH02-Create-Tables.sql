/******************************************************************************/
/*				                                                        	*/
/*	Kroenke and Auer - Database Processing (14th Edition) Chapter 02 		*/
/*										                                    */
/*	The Morgan Importing (MI-CH02) Database - Create Tables                  */
/*							                                                 */
/*	These are the MySQL 5.6 code solutions  						          */
/*								                                             */
/******************************************************************************/


CREATE TABLE ITEM (
		ItemID	            Int				    NOT NULL auto_increment,
		Description		      VarChar(255)	NOT NULL,
		PurchaseDate		    Date			    NOT NULL,
		Store			          Char(50)			NOT NULL,
		City				        Char(35)		  NOT NULL,
		Quantity			      Int		        NOT NULL,
		LocalCurrencyAmount	Numeric(18,2)	NOT NULL,
		ExchangeRate			  Numeric(12,6)	NOT NULL,
		CONSTRAINT	Item_PK	   	 PRIMARY KEY (ItemID)
		);

CREATE TABLE SHIPMENT (
		ShipmentID				    Int				    NOT NULL auto_increment,
		ShipperName		        Char(35)	    NOT NULL,
		ShipperInvoiceNumber	Int				    NOT NULL,
		DepartureDate				  Date			    NULL,
		ArrivalDate				    Date			    NULL,
		InsuredValue			    Numeric(12,2)	NOT NULL,
		CONSTRAINT	Shipment_PK	  PRIMARY KEY (ShipmentID)
		);

 
CREATE TABLE SHIPMENT_ITEM (
		ShipmentID		    Int				    NOT NULL,
		ShipmentItemID		Int				    NOT NULL,
		ItemID		    	  Int				    NOT NULL,
		Value		  	      Numeric(12,2)	NOT NULL,
		CONSTRAINT	ShipmentItem_PK	    PRIMARY KEY(ShipmentID, ShipmentItemID),
		CONSTRAINT	Ship_Item_Ship_FK		    FOREIGN KEY(ShipmentID)
                      REFERENCES SHIPMENT(ShipmentID)
                        ON DELETE CASCADE,
		CONSTRAINT	Ship_Item_Item_FK	FOREIGN KEY(ItemID)
                      REFERENCES ITEM(ItemID)
     		);


