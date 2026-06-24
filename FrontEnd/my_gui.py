from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

class MyGui:

    def __init__(self , root , print_handler):

        self.root = root
        self.root.title('Semantic Image Compression Module')
        self.root.geometry("600x400")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.print_handler = print_handler

        self.imagePath=StringVar(value='Not Selected')

        self._setup_widgets()


        
    
    def pick_image(self):

        file_types = [
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All Files", "*.*"),
        ]

        file_path = filedialog.askopenfilename(
            title="Choose an Image", filetypes=file_types
        )

        # If a file was selected, process and display it
        if file_path:
            self.imagePath.set(file_path)
        else :
            self.imagePath.set('Not Selected')
    
    def _setup_widgets(self  ):
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid(column=0,row=0,sticky='nsew')
        self.frm.columnconfigure(0,weight=1 , uniform='group1')
        self.frm.columnconfigure(1,weight=1,uniform='group1')



        self.imageFrame = ttk.Frame(self.frm )
        self.controlFrame = ttk.Frame(self.frm)

        self.imageFrame.grid(column=1,row=0,sticky='nsew');
        self.controlFrame.grid(column=0 , row=0 , sticky='nsew');
        self.controlFrame.columnconfigure(0,weight=1,uniform='group1')

        
        self.pickbutton=ttk.Button(self.controlFrame,text='Pick Image',command=self.pick_image)
        self.imageLabel=ttk.Label(self.controlFrame,textvariable= self.imagePath)
        self.executeButton=ttk.Button(self.controlFrame , text='Execute',command =self._execute  )
        self.outputDisplay=ScrolledText(self.controlFrame)

        self.pickbutton.pack(fill=X)
        self.imageLabel.pack(fill=X)
        self.executeButton.pack(fill=X)
        self.outputDisplay.configure(state='disabled')
        self.outputDisplay.pack(fill=X,expand=True)
    
    def _update_output(self , output):
        self.outputDisplay.configure(state='normal')
        self.outputDisplay.delete(1.0,END)
        self.outputDisplay.insert(END,output)
        self.outputDisplay.configure(state='disabled')
        self.outputDisplay.see(END)
    
    def _execute(self):
        if self.imagePath.get() == 'Not Selected':
            print('Select a image first')
        else:
            result =  self.print_handler()
            self._update_output(result)
            

