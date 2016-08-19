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
        self.grid()

        self.create_subframes()


    def create_subframes(self):
        
        self.original = tk.Frame(self, bd=1, width=700)
        self.working = tk.Frame(self, bd=1, width=700)
        self.menu = tk.Frame(self, bd=1)

        self.original.grid(row=0, column=0)
        self.working.grid(row=0, column=1)
        self.menu.grid(row=0, column=2)

        self.button = tk.Button(self.menu, text="Load Image", command=self.load_file, width=10)
        self.button.grid(sticky=tk.W)

        self.original.label = tk.Label(self.original, text="Original")
        self.working.label = tk.Label(self.working, text="Working Copy")

        self.original.canvas = tk.Canvas(self.original)
        self.working.canvas = tk.Canvas(self.working)

        self.original.canvas.grid(row = 1, column = 0)
        self.working.canvas.grid(row = 1, column = 0)

        self.original.label.grid(row = 0, column = 0, sticky=tk.W)
        self.working.label.grid(row = 0, column = 0, sticky=tk.W)

    def load_file(self):
        fname = askopenfilename()
        if fname:
            img = Image.open(fname)
            imgtk = ImageTk.PhotoImage(img)
            self.image = imgtk
            self.working.canvas.create_image(0,0, image=imgtk,anchor="nw")
            self.working.canvas.config(scrollregion=self.working.canvas.bbox(tk.ALL))
            self.original.canvas.create_image(200,200, image=imgtk)


if __name__ == "__main__":
    GUI().mainloop()
    
