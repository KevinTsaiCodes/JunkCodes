-- phpMyAdmin SQL Dump
-- version 2.11.7
-- http://www.phpmyadmin.net
--
-- 主機: localhost
-- 建立日期: Oct 02, 2008, 03:33 PM
-- 伺服器版本: 5.0.51
-- PHP 版本: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- 資料庫: `class`
--

-- --------------------------------------------------------

--
-- 資料表格式： `scorelist`
--

CREATE TABLE IF NOT EXISTS `scorelist` (
  `id` tinyint(4) unsigned NOT NULL auto_increment,
  `cID` tinyint(2) unsigned zerofill NOT NULL,
  `course` enum('國文','英文','數學') collate utf8_unicode_ci NOT NULL default '國文',
  `score` tinyint(3) unsigned NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci AUTO_INCREMENT=31 ;

--
-- 列出以下資料庫的數據： `scorelist`
--

INSERT INTO `scorelist` (`id`, `cID`, `course`, `score`) VALUES
(1, 01, '國文', 82),
(2, 02, '國文', 68),
(3, 03, '國文', 78),
(4, 04, '國文', 85),
(5, 05, '國文', 80),
(6, 06, '國文', 76),
(7, 07, '國文', 90),
(8, 08, '國文', 87),
(9, 09, '國文', 78),
(10, 10, '國文', 65),
(11, 01, '英文', 67),
(12, 02, '英文', 87),
(13, 03, '英文', 88),
(14, 04, '英文', 92),
(15, 05, '英文', 55),
(16, 06, '英文', 62),
(17, 07, '英文', 65),
(18, 08, '英文', 40),
(19, 09, '英文', 89),
(20, 10, '英文', 64),
(21, 01, '數學', 87),
(22, 02, '數學', 52),
(23, 03, '數學', 76),
(24, 04, '數學', 56),
(25, 05, '數學', 72),
(26, 06, '數學', 80),
(27, 07, '數學', 38),
(28, 08, '數學', 68),
(29, 09, '數學', 90),
(30, 10, '數學', 61);
