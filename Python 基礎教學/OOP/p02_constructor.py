class Car:
	def __init__(self, speed, color): # init method is a constructor, initialize the member, 建構子
		print('the __init__ is called')
		print(color)
		print(speed)
		self.speed = speed # self.speed is like Java this.speed
ford = Car(200, 'red') # automatic call the method

# ford = Car() <-- error: TypeError: __init__() missing 2 required positional arguments: 'speed' and 'color'
"""
def __init__(self, member_name_1, member_name_2, member_name_3)
"""

"""
這有點類似Java這樣寫法,
public class Student{
	public Student(){ # 這裡等同於Python 的 def __init__ 方法
	
	}
}
"""
