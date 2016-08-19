import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

import sys

class GUI(tk.Frame):

    def __init__(self):
        tk.Frame.__init__(self)
        self.master.title("GUI")

        # Full Windows Size
        self.master.geometry("{0}x{1}+0+0".format(
                        self.master.winfo_screenwidth(), 
                        self.master.winfo_screenheight()))

        self.pack(fill=tk.BOTH,expand=tk.YES)

        self.create_subframes()


    def create_subframes(self):
        
        self.original = tk.Frame(self)
        self.working = tk.Frame(self)
        self.menu = tk.Frame(self, bd=1)

        self.original.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.working.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.YES,  side=tk.RIGHT)

        self.menu.label = tk.Label(self.menu, text="Menu")
        self.menu.label.grid(pady=10, sticky=tk.W+tk.E+tk.N+tk.S)
        self.menu.button = tk.Button(self.menu, text="Load Image", command=self.load_file, width=10)
        self.menu.button.grid(pady=10)
        self.menu.button2 = tk.Button(self.menu, text="Revert", command=self.revert, width=10)
        self.menu.button2.grid(pady=10)

        self.original.label = tk.Label(self.original, text="Original")
        self.working.label = tk.Label(self.working, text="Working Copy")

        self.original.canvas = tk.Canvas(self.original)
        self.working.canvas = tk.Canvas(self.working)

        self.original.label.pack(side=tk.TOP)
        self.working.label.pack()

        self.original.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.working.canvas.pack(fill=tk.BOTH, expand=tk.YES)


    def revert(self):
        self.img = self.img.rotate(180)
        self.imagerotated = ImageTk.PhotoImage(self.img)
        # self.working.canvas.config(image=self.image)
        self.working.canvas.itemconfig(self.id, image = self.imagerotated)

    def load_file(self):
        fname = askopenfilename()
        if fname:
            self.img = Image.open(fname)
            imgtk = ImageTk.PhotoImage(self.img)
            self.image = imgtk
            self.id = self.working.canvas.create_image(0, 0, anchor='nw',image=self.image)
            self.original.canvas.create_image(0, 0, anchor='nw',image=self.image)


if __name__ == "__main__":
    GUI().mainloop()
    
