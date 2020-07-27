import tkinter as tk

win = tk.Tk() # create a window

win.title("window name") # window name

# size
win.geometry("400x200") # 初始開啟大小，寬*高(字串)

win.minsize(400,200) # 相反: object_name.maxsize(width,height)

win.resizable(False,False) # 可縮放性, 1 = True, 0 = False

win.iconbitmap() # create window's icon(.ico)
# Example, win.iconbitmap("~img/ico/xxx.ico")

# Color
win.config(background="red")

win.mainloop()
