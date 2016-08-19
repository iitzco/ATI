import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
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
        self.studio = tk.Frame(self)
        self.menu = tk.Frame(self, bd=1)

        self.original.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.studio.pack(fill=tk.BOTH, expand=tk.YES, side=tk.LEFT)
        self.menu.pack(fill=tk.Y, expand=tk.YES,  side=tk.RIGHT)

        self.menu.label = tk.Label(self.menu, text="Menu")
        self.menu.label.grid(pady=10)
        self.menu.button = tk.Button(self.menu, text="Load Image", command=self.load_file, width=10)
        self.menu.button.grid(pady=10)
        self.menu.button2 = tk.Button(self.menu, text="Revert", command=self.revert, width=10)
        self.menu.button2.grid(pady=10)
        self.menu.button3 = tk.Button(self.menu, text="Save", command=self.save_file, width=10)
        self.menu.button3.grid(pady=10)

        self.menu.zoom = tk.Canvas(self.menu, height=150, width=150)
        self.menu.zoom.grid()
        self.menu.zoom.create_text(10,10, anchor='nw',text="Click Image to zoom")

        self.menu.slider = tk.Scale(self.menu, from_=0, to=255, orient=tk.HORIZONTAL)
        self.menu.slider.grid(pady=10)

        self.original.label = tk.Label(self.original, text="Original")
        self.studio.label = tk.Label(self.studio, text="Studio")

        self.original.canvas = tk.Canvas(self.original)
        self.studio.canvas = tk.Canvas(self.studio)

        self.original.label.pack(side=tk.TOP)
        self.studio.label.pack()

        self.original.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.studio.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.studio.canvas.bind("<B1-Motion>", self.zoom)
        self.studio.canvas.bind("<Button-1>", self.zoom)

    def revert(self):
        self.img = self.img.rotate(180)
        self.imagerotated = ImageTk.PhotoImage(self.img)
        self.studio.canvas.itemconfig(self.id, image = self.imagerotated)

    def load_file(self):
        fname = askopenfilename()
        if fname:
            self.img = Image.open(fname)
            imgtk = ImageTk.PhotoImage(self.img)
            self.image = imgtk
            self.id = self.studio.canvas.create_image(0, 0, anchor='nw',image=self.image)
            self.original.canvas.create_image(0, 0, anchor='nw',image=self.image)

    def save_file(self):
        fname = asksaveasfilename()
        if fname:
            self.img.save(fname)

    def zoom(self, event):
        self.imgCropped = self.img.crop((event.x-20, event.y-20, event.x+20, event.y+20)).resize((150,150))
        self.imagetk = ImageTk.PhotoImage(self.imgCropped)
        self.menu.zoom.create_image(0, 0, anchor='nw',image=self.imagetk)


if __name__ == "__main__":
    GUI().mainloop()
    
