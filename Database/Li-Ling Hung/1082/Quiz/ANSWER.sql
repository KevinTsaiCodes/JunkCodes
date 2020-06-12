CREATE TABLE SELLER(
	ShoesNumber INT PRIMARY KEY NOT NULL,
	ShoesName CHAR(50) NOT NULL,
	SalePrice INT NOT NULL,
	ReOrderQuantity INT NOT NULL,
	QuantityOnHand INT NOT NULL
);
CREATE TABLE MANUFACTURER(
	ManufacturerName CHAR(20) PRIMARY KEY NOT NULL,
	City CHAR(50) NOT NULL,
	Country CHAR(50) NOT NULL,
	Volume INT NOT NULL
);
CREATE TABLE QUOTATION(
	Price INT NOT NULL,
	ShoesNumber INT,
	ManufacturerName CHAR(20)
	CONSTRAINT ShoesNumber FOREIGN KEY(ShoesNumber)
	REFERENCES SELLER(ShoesNumber),
	CONSTRAINT ManufacturerName FOREIGN KEY(ManufacturerName)
	REFERENCES MANUFACTURER(ManufacturerName)
);