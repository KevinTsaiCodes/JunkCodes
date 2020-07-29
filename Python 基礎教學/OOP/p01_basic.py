class Car:
	pass # pass means it is an empty class or method

ford = Car() # ford is an object
honda = Car()
audi = Car()

ford.speed = 200 # speed is a attribute
honda.color = 'red'


print(ford.speed,
	'\n',honda.color)

ford.speed = 500 # will change the attribute
honda.color = 'blue'

print(ford.speed,
	'\n',honda.color)
